/**
 * v5.7.0: Deploy Control Panel - CCC Convergence-Aware Deployment Management
 *
 * Integrates the K-Kristensen Convergence Readiness framework from the
 * "Cosmic Arcology Mission" paper into the 5-server HA deployment pipeline.
 *
 * Cosmic phases map to deployment stages:
 *   Isolation → Alpha + Delta deploying in parallel
 *   Convergence → Gamma verifying, syncing with Beta
 *   Aeon Transition → Beta deploying (conformal boundary crossing)
 *   Harmony → All 5 servers unified, same version, synced
 *
 * Pipeline: Alpha+Delta (parallel) → Gamma (verify) → Beta (primary)
 * Only visible to the master wallet (FOUNDER_WALLET).
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, X, Server, Activity, CheckCircle, XCircle,
  AlertTriangle, RefreshCw, Rocket, RotateCcw, Wifi, WifiOff,
  Clock, Layers, Users, Zap, Globe, Radio, ArrowRight,
  Database, TrendingUp, MonitorSmartphone, Timer, DollarSign, Settings, Save,
  Landmark, CreditCard, FileText, Send, Eye, Trash2, BadgeCheck, Banknote,
  ChevronDown, ChevronRight, Copy, Terminal, AlertCircle, Wallet,
  Key, Cpu, Lock, Fingerprint, Hash, Award, BarChart3
} from 'lucide-react';
import { getConnectionInfo } from '../services/api';

const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';

/** Server-side detailed sync status (from /api/v1/sync/detailed) */
interface DetailedSyncStatus {
  sync_mode: string;
  local_height: number;
  network_height: number;
  gap: number;
  total_chunks: number;
  completed_chunks: number;
  in_flight: number;
  queued: number;
  chunk_progress_pct: number;
  blocks_per_second: number;
  bytes_downloaded_mb: number;
  compression_ratio: number;
  active_streams: number;
  failed_chunks: number;
  retried_chunks: number;
  peer_count: number;
  best_peer_height: number;
  is_fully_synced: boolean;
  eta_seconds: number | null;
  // v8.2.8: Apollo subsystem metrics
  apollo_kalman_bandwidth_mbps?: number;
  apollo_kalman_latency_ms?: number;
  apollo_kalman_confidence?: number;
  apollo_kalman_optimal_chunk_kb?: number;
  apollo_kalman_loss_pct?: number;
  apollo_kalman_timeout_ms?: number;
  apollo_kalman_concurrency?: number;
  apollo_kalman_jitter_ms?: number;
  apollo_kalman_samples?: number;
  apollo_pid_target_bps?: number;
  apollo_pid_current_bps?: number;
  apollo_pid_error?: number;
  apollo_pid_kp?: number;
  apollo_pid_ki?: number;
  apollo_pid_kd?: number;
  apollo_peers_tracked?: number;
  apollo_gravity_best_peer?: string;
  apollo_gravity_best_heat?: number;
}

interface NodeStatus {
  name: string;
  url: string;
  online: boolean;
  version: string;
  height: number;
  network_height: number;
  peers: number;
  uptime_secs: number;
  status: string;
  sync_details?: DetailedSyncStatus;
}

/** Computed sync metrics (client-side) */
interface SyncMetrics {
  speed: number;        // blocks/sec
  eta_secs: number;     // seconds to fully synced (-1 = synced, 0 = unknown)
  sync_pct: number;     // 0-100
  gap: number;          // blocks behind network
}

interface DeployStatus {
  alpha: NodeStatus;
  beta: NodeStatus;
  gamma: NodeStatus;
  delta: NodeStatus;
  epsilon: NodeStatus;
  height_delta: number;
  versions_match: boolean;
}

// v1.0.2: Mining capacity metrics per server
interface MiningCapacityLocal {
  queue_used: number;
  queue_capacity: number;
  queue_pct: number;
  hashrate_hs: number;
  active_miners: number;
  solutions_submitted: number;
  solutions_accepted: number;
  acceptance_pct: number;
  is_healthy: boolean;
  last_solution_secs_ago: number;
  shard_count: number;
}

interface MiningCapacityAll {
  beta: MiningCapacityLocal | null;
  gamma: MiningCapacityLocal | null;
  delta: MiningCapacityLocal | null;
  epsilon: MiningCapacityLocal | null;
  alpha: MiningCapacityLocal | null;
}

// v9.0.6: Caddy reverse proxy metrics (replaces nginx stats)
interface CaddyRequestsByStatus {
  ok_2xx: number;
  redirect_3xx: number;
  client_err_4xx: number;
  server_err_5xx: number;
  websocket_101: number;
}

interface CaddyUpstream {
  address: string;
  healthy: boolean;
}

interface CaddyStats {
  total_requests: number;
  requests_by_status: CaddyRequestsByStatus;
  requests_per_second: number;
  avg_response_ms: number;
  p99_response_ms: number;
  goroutines: number;
  memory_mb: number;
  upstreams: CaddyUpstream[];
  server_name: string;
  last_reload: number;
  online: boolean;
}

interface CaddyStatsAll {
  epsilon: CaddyStats | null;
}

// v9.2.0: q-flux reverse proxy stats (worker-per-core TLS proxy)
interface FluxBackendHealth {
  addr: string;
  healthy: boolean;
  failures: number;
  last_check_ms_ago: number;
}

interface FluxClusterInfo {
  enabled: boolean;
  local_backends: FluxBackendHealth[];
  cluster_peers: FluxBackendHealth[];
}

interface FluxStats {
  version: string;
  worker_count: number;
  uptime_secs: number;
  active_connections: number;
  total_connections: number;
  tls_handshakes: number;
  tls_handshake_failures: number;
  total_requests: number;
  requests_2xx: number;
  requests_4xx: number;
  requests_5xx: number;
  upstream_active: number;
  upstream_connect_failures: number;
  upstream_timeouts: number;
  rate_limited: number;
  active_websockets: number;
  websocket_upgrades: number;
  bytes_received: number;
  bytes_sent: number;
  tls_reload_count: number;
  h2_connections: number;
  h2_streams_opened: number;
  h2_streams_closed: number;
  cluster?: FluxClusterInfo;
  online: boolean;
  requests_per_second: number;
  error_rate_pct: number;
}

// v9.0.6: Decentralization Index metrics — sqrt scaling, EMA smoothing, wealth Gini, Shannon entropy
interface DecentralizationMetrics {
  unique_wallets: number;
  total_workers: number;
  top_miner_pct: number;
  top3_miners_pct: number;
  nakamoto_coefficient: number;
  gini_coefficient: number;
  hhi: number;
  node_count: number;
  peer_count: number;
  wealth_gini: number;
  entropy_score: number;
  infrastructure_nodes: number;
  community_nodes: number;
  decentralization_index_raw: number;
  decentralization_index: number;
  grade: string;
}

interface VerifyEvent {
  step: string;
  status: string;
  message: string;
  timestamp: number;
}

// CCC Convergence Types (from Cosmic Arcology Mission paper)
interface NodeKMetrics {
  name: string;
  genetic_stability: number;
  quantum_coherence: number;
  thermodynamic_efficiency: number;
  information_density: number;
  network_resilience: number;
  k_parameter: number;
}

interface ConvergenceStatus {
  cosmic_phase: any; // DeployCosmicPhase enum
  nodes: NodeKMetrics[];
  collective_k: number;
  predicted_outcome: any;
  convergence_safe: boolean;
  gardener_wisdom: string;
  phase_transition_eta: number | null;
}

interface DevFeeStatus {
  fee_bps: number;
  fee_percent: string;
  founder_wallet: string;
  founder_balance_qug: number;
  total_dev_fees_collected: number;
  total_mining_rewards: number;
  actual_fee_ratio: number;
  expected_fee_ratio: number;
  fee_verified: boolean;
  blocks_processed: number;
  today_dev_fee_qug: number;
  today_expected_dev_fee_qug: number;
}

/** Extract the phase name from the DeployCosmicPhase tagged enum */
function getPhaseInfo(phase: any): { name: string; icon: string; color: string; bgGlow: string } {
  if (!phase) return { name: 'Unknown', icon: '?', color: 'text-slate-400', bgGlow: '' };
  if (phase.Isolation !== undefined) return {
    name: 'Isolation',
    icon: '\u{1F30C}', // galaxy emoji
    color: 'text-purple-300',
    bgGlow: 'shadow-[0_0_20px_rgba(168,85,247,0.3)]',
  };
  if (phase.Convergence !== undefined) return {
    name: 'Convergence',
    icon: '\u{1F300}', // cyclone
    color: 'text-blue-300',
    bgGlow: 'shadow-[0_0_20px_rgba(59,130,246,0.3)]',
  };
  if (phase.AeonTransition !== undefined) return {
    name: 'Aeon Transition',
    icon: '\u{1F31F}', // star
    color: 'text-amber-300',
    bgGlow: 'shadow-[0_0_20px_rgba(245,158,11,0.3)]',
  };
  if (phase.Harmony !== undefined) return {
    name: 'Harmony',
    icon: '\u262E\uFE0F', // peace
    color: 'text-emerald-300',
    bgGlow: 'shadow-[0_0_20px_rgba(16,185,129,0.3)]',
  };
  return { name: 'Unknown', icon: '?', color: 'text-slate-400', bgGlow: '' };
}

/** Get convergence outcome name and styling */
function getOutcomeInfo(outcome: any): { name: string; color: string; desc: string } {
  if (!outcome) return { name: 'Unknown', color: 'text-slate-400', desc: '' };
  if (outcome.Communion !== undefined) return {
    name: 'Communion',
    color: 'text-emerald-400',
    desc: `Peaceful merger (synergy +${((outcome.Communion.synergy_bonus || 0) * 100).toFixed(0)}%)`,
  };
  if (outcome.Observation !== undefined) return {
    name: 'Observation',
    color: 'text-blue-400',
    desc: 'Safe limited contact',
  };
  if (outcome.Competition !== undefined) return {
    name: 'Competition',
    color: 'text-amber-400',
    desc: 'Resource equilibrium',
  };
  if (outcome.Conflict !== undefined) return {
    name: 'Conflict',
    color: 'text-red-400',
    desc: `Risk: ${((outcome.Conflict.risk || 0) * 100).toFixed(0)}%`,
  };
  if (outcome.Absorption !== undefined) return {
    name: 'Absorption',
    color: 'text-red-500',
    desc: 'Rollback required',
  };
  return { name: 'Unknown', color: 'text-slate-400', desc: '' };
}

/** Rich tooltip wrapper - shows educational popover on hover */
function KTooltip({ children, text, wide }: { children: React.ReactNode; text: string; wide?: boolean }) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState<'above' | 'below'>('above');
  const ref = useRef<HTMLDivElement>(null);

  const handleEnter = () => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setPos(rect.top < 220 ? 'below' : 'above');
    }
    setShow(true);
  };

  return (
    <div className="relative inline-block" ref={ref}
      onMouseEnter={handleEnter} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <div className={`absolute z-[9999] ${wide ? 'w-72' : 'w-60'} px-3 py-2.5 rounded-lg
          bg-slate-900/95 border border-amber-500/30 shadow-xl shadow-amber-900/20 backdrop-blur-sm
          text-[10px] leading-relaxed text-amber-100/80 pointer-events-none
          ${pos === 'above' ? 'bottom-full mb-2' : 'top-full mt-2'} left-1/2 -translate-x-1/2`}>
          <div className="whitespace-pre-line">{text}</div>
          <div className={`absolute left-1/2 -translate-x-1/2 w-2 h-2 rotate-45 bg-slate-900/95 border-amber-500/30
            ${pos === 'above' ? 'top-full -mt-1 border-r border-b' : 'bottom-full -mb-1 border-l border-t'}`} />
        </div>
      )}
    </div>
  );
}

/** K-Parameter gauge mini-component */
function KGauge({ value, label, size = 'sm' }: { value: number; label: string; size?: 'sm' | 'lg' }) {
  const percent = Math.min(value * 100, 100);
  const color = value > 0.9 ? '#10b981' : value > 0.7 ? '#3b82f6' : value > 0.5 ? '#f59e0b' : '#ef4444';
  const radius = size === 'lg' ? 28 : 16;
  const stroke = size === 'lg' ? 4 : 3;
  const circ = 2 * Math.PI * radius;
  const dashOffset = circ * (1 - value);

  return (
    <div className="flex flex-col items-center gap-0.5">
      <svg width={(radius + stroke) * 2} height={(radius + stroke) * 2} className="transform -rotate-90">
        <circle cx={radius + stroke} cy={radius + stroke} r={radius} fill="none"
          stroke="rgba(255,255,255,0.08)" strokeWidth={stroke} />
        <circle cx={radius + stroke} cy={radius + stroke} r={radius} fill="none"
          stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={dashOffset}
          style={{ transition: 'stroke-dashoffset 1s ease' }} />
      </svg>
      <span className={`${size === 'lg' ? 'text-sm font-bold' : 'text-[10px] font-medium'}`}
        style={{ color, marginTop: size === 'lg' ? -((radius * 2) / 2 + 8) : -(radius + 4) }}>
        {value.toFixed(2)}
      </span>
      <span className="text-[9px] text-amber-200/50 mt-0.5">{label}</span>
    </div>
  );
}

/**
 * Educational info panel for the K-parameter orbital visualization.
 * Two-tier explanation: formal (university) and intuitive (high school).
 * Shown on hover / click over the orbital canvas.
 */
function KOrbitalInfoPanel({ phase, kValue, show, onClose }: {
  phase: string; kValue: number; show: boolean; onClose: () => void;
}) {
  if (!show) return null;

  const phaseLabel = phase === 'critical' ? 'Critical' : phase === 'approaching' ? 'Approaching' : 'Stable';
  const phaseColor = phase === 'critical' ? '#ef4444' : phase === 'approaching' ? '#f59e0b' : '#10b981';
  const phaseEmoji = phase === 'critical' ? '\u26A0' : phase === 'approaching' ? '\u26A1' : '\u2714';

  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="absolute inset-0 z-20 overflow-y-auto rounded-xl"
      style={{
        background: 'rgba(2, 6, 23, 0.97)',
        backdropFilter: 'blur(16px)',
        border: `1px solid ${phaseColor}33`,
      }}
      onClick={onClose}
    >
      <div className="p-4 space-y-3" onClick={e => e.stopPropagation()}>
        {/* Title bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: phaseColor }} />
            <span className="text-[11px] font-bold tracking-widest uppercase" style={{ color: phaseColor }}>
              K-Parameter Field Guide
            </span>
          </div>
          <button onClick={onClose} className="p-1 rounded-md hover:bg-white/10 transition-colors">
            <X className="w-3 h-3 text-slate-400" />
          </button>
        </div>

        {/* Current reading */}
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg" style={{ background: `${phaseColor}0D`, border: `1px solid ${phaseColor}22` }}>
          <span className="text-lg font-bold font-mono" style={{ color: phaseColor }}>{kValue.toFixed(2)}</span>
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: phaseColor }}>
            {phaseEmoji} {phaseLabel}
          </span>
          <span className="text-[9px] text-slate-500 ml-auto font-mono">K = 2\u03C0 \u221A(\u0394H\u00B7\u0394s\u00B7\u210F) / \u03C4</span>
        </div>

        {/* University explanation */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5">
            <div className="px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wider bg-cyan-500/15 text-cyan-300 border border-cyan-500/20">
              FORMAL
            </div>
            <span className="text-[9px] text-slate-500 italic">University / Graduate Level</span>
          </div>
          <div className="text-[10px] leading-[1.6] text-slate-300 px-2 py-2 rounded-lg bg-slate-800/40 border border-slate-700/30">
            <p className="mb-2">
              The <span className="font-semibold text-cyan-300">K-parameter</span> is a dimensionless
              composite metric derived from five orthogonal health observables of a distributed
              consensus network. Formally:
            </p>
            <p className="font-mono text-[9px] text-center text-cyan-400/80 py-1">
              K = 2\u03C0 \u221A(\u0394H \u00B7 \u0394s \u00B7 \u210F) / \u03C4
            </p>
            <p className="mb-2">
              where <span className="text-cyan-400">\u0394H</span> captures Hamiltonian energy divergence
              in the DAG consensus,{' '}
              <span className="text-cyan-400">\u0394s</span> is the entropy production rate across
              the peer mesh, <span className="text-cyan-400">\u210F</span> represents the reduced
              Planck-analog (minimum quantum of state agreement), and{' '}
              <span className="text-cyan-400">\u03C4</span> is the gossip propagation time constant.
            </p>
            <p className="mb-1.5">
              The five factor components \u2014{' '}
              <span className="text-emerald-400 font-semibold">G</span> (Genetic Stability),{' '}
              <span className="text-cyan-400 font-semibold">Q</span> (Quantum Coherence),{' '}
              <span className="text-orange-400 font-semibold">T</span> (Thermodynamic Efficiency),{' '}
              <span className="text-blue-400 font-semibold">I</span> (Information Density),{' '}
              <span className="text-purple-400 font-semibold">R</span> (Network Resilience)
              {' '}\u2014 are visualized as orbiting particles. Each traces an elliptical path whose
              orbital velocity scales with system stress: higher K pushes faster orbits, mirroring
              increased phase-space exploration under entropic pressure.
            </p>
            <p>
              The <span className="font-semibold" style={{ color: phaseColor }}>phase regime</span> is
              determined by critical thresholds: K &lt; 5 ={' '}
              <span className="text-emerald-400">ordered (stable)</span>,
              5 \u2264 K &lt; 10 ={' '}
              <span className="text-amber-400">approaching (meta-stable)</span>,
              K \u2265 10 ={' '}
              <span className="text-red-400">critical (disordered)</span>.
              These boundaries correspond to phase transitions in the consensus Hamiltonian landscape.
            </p>
          </div>
        </div>

        {/* High school explanation */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5">
            <div className="px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wider bg-amber-500/15 text-amber-300 border border-amber-500/20">
              INTUITIVE
            </div>
            <span className="text-[9px] text-slate-500 italic">Plain English / High School</span>
          </div>
          <div className="text-[10px] leading-[1.6] text-slate-300 px-2 py-2 rounded-lg bg-slate-800/40 border border-slate-700/30">
            <p className="mb-2">
              Think of K as the network's <span className="font-semibold text-amber-300">temperature reading</span>.
              When you're healthy, your body temperature is ~37\u00B0C. Too high means fever. Too low means trouble.
              The K-parameter works the same way for this blockchain network.
            </p>
            <p className="mb-2">
              <span className="font-semibold text-emerald-400">K near 0 = perfectly healthy.</span>{' '}
              All servers are running the same software, agree on the same block history, aren't overworked,
              and can talk to each other easily. Like a well-rehearsed orchestra playing in sync.
            </p>
            <p className="mb-2">
              <span className="font-semibold text-amber-400">K between 5\u201310 = getting stressed.</span>{' '}
              Something is off. Maybe one server is lagging behind, or the network is getting congested.
              Like a band where the drummer is slightly off-beat \u2014 still playable, but you notice it.
            </p>
            <p className="mb-2">
              <span className="font-semibold text-red-400">K above 10 = something is wrong.</span>{' '}
              Servers disagree, connections are dropping, or a node is overwhelmed. Like a traffic jam:
              everyone's honking but nobody's moving. The system tunes itself harder to compensate.
            </p>
            <p className="mb-1.5">
              The five orbiting dots represent the five things being measured:
            </p>
            <div className="grid grid-cols-1 gap-1 pl-1 mb-1.5">
              <div><span className="font-bold text-emerald-400">G</span><span className="text-slate-400"> \u2014 Are all servers running the same version? (like everyone reading the same textbook)</span></div>
              <div><span className="font-bold text-cyan-400">Q</span><span className="text-slate-400"> \u2014 Are they at the same block height? (like clocks being in sync)</span></div>
              <div><span className="font-bold text-orange-400">T</span><span className="text-slate-400"> \u2014 Are they using resources efficiently? (CPU, RAM \u2014 like a car's fuel efficiency)</span></div>
              <div><span className="font-bold text-blue-400">I</span><span className="text-slate-400"> \u2014 Is meaningful data flowing through? (like how much of a conversation is signal vs noise)</span></div>
              <div><span className="font-bold text-purple-400">R</span><span className="text-slate-400"> \u2014 Can they handle failures gracefully? (like a power grid rerouting around a downed line)</span></div>
            </div>
            <p>
              When K is low, the dots orbit slowly and calmly. When K rises, they speed up and the core
              glows brighter \u2014 a visual alarm that the system is under pressure. The waveform at the bottom
              shows K's history over time, so you can spot trends.
            </p>
          </div>
        </div>

        {/* Visual legend */}
        <div className="px-2 py-1.5 rounded-lg bg-slate-800/30 border border-slate-700/20">
          <div className="text-[8px] font-bold text-slate-500 uppercase tracking-wider mb-1">Visual Legend</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[9px] text-slate-400">
            <div><span className="text-white/60">\u25CF Core glow</span> \u2014 size & color = current K severity</div>
            <div><span className="text-white/60">\u25CB Orbit rings</span> \u2014 breathe with system stress</div>
            <div><span className="text-white/60">\u2500 Dashed arcs</span> \u2014 K=5 and K=10 phase thresholds</div>
            <div><span className="text-white/60">\u223F Waveform</span> \u2014 K history over recent rounds</div>
            <div><span className="text-white/60">\u2022 Particles</span> \u2014 G Q T I R health factors</div>
            <div><span className="text-white/60">\u2606 Trail sparks</span> \u2014 data emission from each factor</div>
          </div>
        </div>

        <div className="text-center text-[8px] text-slate-600 pt-0.5">
          Click anywhere to close
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Animated orbital K-parameter visualization.
 * The K value drives a central pulsing core surrounded by orbiting factor particles.
 * Phase transitions trigger dramatic visual shifts.
 */
function KOrbitalViz({ kValue, phase, history }: {
  kValue: number;
  phase: string;
  history: Array<{ k: number; phase: string; ts: number }>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const prevKRef = useRef(kValue);
  const transitionRef = useRef(0); // 0-1 for phase transition flash

  // Smooth interpolation target
  const targetK = useRef(kValue);
  const currentK = useRef(kValue);

  useEffect(() => {
    targetK.current = kValue;
    // Trigger transition flash when phase changes
    if (kValue !== prevKRef.current) {
      const oldPhase = prevKRef.current < 5 ? 'stable' : prevKRef.current < 10 ? 'approaching' : 'critical';
      const newPhase = kValue < 5 ? 'stable' : kValue < 10 ? 'approaching' : 'critical';
      if (oldPhase !== newPhase) transitionRef.current = 1.0;
      prevKRef.current = kValue;
    }
  }, [kValue]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybeCtx = canvas.getContext('2d');
    if (!maybeCtx) return;
    const ctx: CanvasRenderingContext2D = maybeCtx;

    // Hi-DPI support
    const dpr = window.devicePixelRatio || 1;
    const W = 320;
    const H = 200;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    const cx = W / 2;
    const cy = H / 2;

    // Factor particles
    const factors = [
      { letter: 'G', color: '#10b981', orbit: 55, speed: 0.8, offset: 0 },
      { letter: 'Q', color: '#06b6d4', orbit: 55, speed: 1.1, offset: Math.PI * 0.4 },
      { letter: 'T', color: '#f97316', orbit: 55, speed: 0.6, offset: Math.PI * 0.8 },
      { letter: 'I', color: '#3b82f6', orbit: 55, speed: 0.9, offset: Math.PI * 1.2 },
      { letter: 'R', color: '#a855f7', orbit: 55, speed: 0.7, offset: Math.PI * 1.6 },
    ];

    // History trail particles
    const trails: Array<{ x: number; y: number; vx: number; vy: number; life: number; color: string }> = [];

    let t = 0;

    const phaseColor = (p: string) =>
      p === 'critical' ? '#ef4444' : p === 'approaching' ? '#f59e0b' : '#10b981';
    const phaseGlow = (p: string) =>
      p === 'critical' ? 'rgba(239, 68, 68,' : p === 'approaching' ? 'rgba(245, 158, 11,' : 'rgba(16, 185, 129,';

    function draw() {
      t += 0.016;
      // Smooth K interpolation
      currentK.current += (targetK.current - currentK.current) * 0.05;
      const k = currentK.current;
      const p = k < 5 ? 'stable' : k < 10 ? 'approaching' : 'critical';
      const kNorm = Math.min(k / 15, 1); // 0-1 normalized

      // Decay transition flash
      if (transitionRef.current > 0) transitionRef.current *= 0.95;

      ctx.clearRect(0, 0, W, H);

      // Background radial gradient (phase-colored)
      const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 120);
      bgGrad.addColorStop(0, `${phaseGlow(p)}0.08)`);
      bgGrad.addColorStop(0.5, `${phaseGlow(p)}0.02)`);
      bgGrad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, W, H);

      // Phase transition flash
      if (transitionRef.current > 0.01) {
        const flashGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 150);
        flashGrad.addColorStop(0, `rgba(255,255,255,${transitionRef.current * 0.5})`);
        flashGrad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = flashGrad;
        ctx.fillRect(0, 0, W, H);
      }

      // Orbit ring(s) — breathing effect based on K
      const breathScale = 1 + Math.sin(t * 2) * 0.03 * (1 + kNorm);
      const orbitR = 55 * breathScale;

      // Outer orbit ring
      ctx.beginPath();
      ctx.arc(cx, cy, orbitR, 0, Math.PI * 2);
      ctx.strokeStyle = `${phaseGlow(p)}${(0.15 + kNorm * 0.15).toFixed(2)})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 6]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Inner ring (secondary orbit at 35px)
      ctx.beginPath();
      ctx.arc(cx, cy, 35 * breathScale, 0, Math.PI * 2);
      ctx.strokeStyle = `${phaseGlow(p)}0.08)`;
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Phase threshold arcs (K=5 and K=10 as partial rings)
      const drawThresholdArc = (threshold: number, color: string, label: string) => {
        const angle = (threshold / 15) * Math.PI * 2 - Math.PI / 2;
        const r = 80;
        ctx.beginPath();
        ctx.arc(cx, cy, r, angle - 0.15, angle + 0.15);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        // Tiny label
        const lx = cx + Math.cos(angle) * (r + 8);
        const ly = cy + Math.sin(angle) * (r + 8);
        ctx.font = '7px monospace';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, lx, ly);
      };
      drawThresholdArc(5, 'rgba(245, 158, 11, 0.5)', 'K=5');
      drawThresholdArc(10, 'rgba(239, 68, 68, 0.5)', 'K=10');

      // Core glow (pulsing, size proportional to K)
      const coreR = 16 + kNorm * 10 + Math.sin(t * 3) * 2;
      const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 2);
      coreGlow.addColorStop(0, `${phaseGlow(p)}0.6)`);
      coreGlow.addColorStop(0.4, `${phaseGlow(p)}0.15)`);
      coreGlow.addColorStop(1, `${phaseGlow(p)}0)`);
      ctx.fillStyle = coreGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR * 2, 0, Math.PI * 2);
      ctx.fill();

      // Core solid
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
      coreGrad.addColorStop(0, 'rgba(255,255,255,0.25)');
      coreGrad.addColorStop(0.5, phaseColor(p));
      coreGrad.addColorStop(1, `${phaseGlow(p)}0.3)`);
      ctx.fillStyle = coreGrad;
      ctx.fill();

      // K value text in core
      ctx.font = 'bold 14px monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.95)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(k.toFixed(2), cx, cy - 2);
      ctx.font = '7px monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.fillText('K-VALUE', cx, cy + 10);

      // Orbiting factor particles
      const speedMult = p === 'critical' ? 2.0 : p === 'approaching' ? 1.3 : 1.0;
      factors.forEach((f, i) => {
        const angle = t * f.speed * speedMult + f.offset;
        // Elliptical orbit (wider horizontally)
        const fx = cx + Math.cos(angle) * orbitR * 1.3;
        const fy = cy + Math.sin(angle) * orbitR * 0.7;

        // Trail spawn
        if (Math.random() < 0.3) {
          trails.push({
            x: fx, y: fy,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            life: 1.0,
            color: f.color,
          });
        }

        // Connection line to core
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(fx, fy);
        ctx.strokeStyle = `${f.color}33`;
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Particle glow
        const pGlow = ctx.createRadialGradient(fx, fy, 0, fx, fy, 12);
        pGlow.addColorStop(0, `${f.color}88`);
        pGlow.addColorStop(1, `${f.color}00`);
        ctx.fillStyle = pGlow;
        ctx.beginPath();
        ctx.arc(fx, fy, 12, 0, Math.PI * 2);
        ctx.fill();

        // Particle solid
        ctx.beginPath();
        ctx.arc(fx, fy, 6, 0, Math.PI * 2);
        ctx.fillStyle = f.color;
        ctx.fill();

        // Letter
        ctx.font = 'bold 8px monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(f.letter, fx, fy);
      });

      // Update and draw trail particles
      for (let i = trails.length - 1; i >= 0; i--) {
        const tr = trails[i];
        tr.x += tr.vx;
        tr.y += tr.vy;
        tr.life -= 0.02;
        if (tr.life <= 0) { trails.splice(i, 1); continue; }
        ctx.beginPath();
        ctx.arc(tr.x, tr.y, 1.5 * tr.life, 0, Math.PI * 2);
        ctx.fillStyle = `${tr.color}${Math.floor(tr.life * 60).toString(16).padStart(2, '0')}`;
        ctx.fill();
      }
      // Cap trail particles to prevent memory growth
      if (trails.length > 200) trails.splice(0, trails.length - 200);

      // History waveform at bottom
      if (history.length > 1) {
        const waveY = H - 18;
        const waveH = 14;
        const step = (W - 40) / Math.max(history.length - 1, 1);
        ctx.beginPath();
        ctx.moveTo(20, waveY);
        history.forEach((pt, i) => {
          const x = 20 + i * step;
          const y = waveY - (pt.k / 15) * waveH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = `${phaseGlow(p)}0.5)`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Dot at current position
        const lastPt = history[history.length - 1];
        const lastX = 20 + (history.length - 1) * step;
        const lastY = waveY - (lastPt.k / 15) * waveH;
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fillStyle = phaseColor(p);
        ctx.fill();
      }

      // Formula watermark
      ctx.font = '8px monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.textAlign = 'center';
      ctx.fillText('K = 2\u03C0 \u221A(\u0394H \u00B7 \u0394s \u00B7 \u210F) / \u03C4', cx, H - 4);

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [history]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: 320, height: 200 }}
      className="rounded-xl mx-auto block"
    />
  );
}

/** K-metrics breakdown bar for a single node */
function KMetricsBar({ metrics }: { metrics: NodeKMetrics }) {
  const factors = [
    { key: 'G', value: metrics.genetic_stability, label: 'Genetic Stability', exp: 0.25,
      tooltip: (v: number) => {
        const pct = (v * 100).toFixed(0);
        const weighted = Math.pow(v, 0.25);
        const grade = v > 0.9 ? 'Excellent — all servers share the same DNA' : v > 0.7 ? 'Good — minor version drift, but compatible' : v > 0.5 ? 'Warning — version mismatch detected' : 'Critical — servers running incompatible versions';
        return `G — Genetic Stability: ${v.toFixed(4)} (${pct}%, weight 25%)\nWeighted contribution: ${v.toFixed(4)}^0.25 = ${weighted.toFixed(4)}\n\n${grade}\n\nWhat it measures: Are all servers running the same software version?\n\nHow it's computed:\n  • Compares binary commit hash across all connected peers\n  • G = 1.0 when all nodes match exactly (same version, same consensus rules)\n  • G degrades proportionally to version distance (patch diff = small penalty, major diff = large penalty)\n  • A single node on a different major version can drop G below 0.5\n\nWhy highest weight (25%): Version mismatch is the #1 cause of network splits. If one node accepts a block that another rejects (different validation rules), the chain forks — users on different forks see different balances. Everything else is pointless if servers can't agree on the rules.\n\nReal example: If Beta runs v9.5.0 and Gamma runs v9.4.1, G ≈ 0.85 (compatible but different). If Beta runs v9.5.0 and Gamma runs v8.0.0, G ≈ 0.20 (consensus rules likely incompatible).`;
      }
    },
    { key: 'Q', value: metrics.quantum_coherence, label: 'Quantum Coherence', exp: 0.20,
      tooltip: (v: number) => {
        const pct = (v * 100).toFixed(0);
        const weighted = Math.pow(v, 0.20);
        const grade = v > 0.9 ? 'Excellent — node is in perfect lockstep with the network' : v > 0.7 ? 'Good — minor height lag, catching up' : v > 0.5 ? 'Warning — falling behind the network' : 'Critical — node may be on a different fork';
        return `Q — Quantum Coherence: ${v.toFixed(4)} (${pct}%, weight 20%)\nWeighted contribution: ${v.toFixed(4)}^0.20 = ${weighted.toFixed(4)}\n\n${grade}\n\nWhat it measures: Is this node's blockchain state synchronized with the network?\n\nHow it's computed:\n  • Q = 1.0 - (height_gap / max_tolerable_gap)\n  • height_gap = network_tip - node_height\n  • max_tolerable_gap ≈ 100 blocks (configurable)\n  • Also factors in: block hash agreement at shared heights, transaction pool overlap\n\nScoring examples:\n  • Gap = 0 blocks → Q = 1.00 (perfect sync)\n  • Gap = 2 blocks → Q ≈ 0.98 (normal propagation delay)\n  • Gap = 50 blocks → Q ≈ 0.50 (node struggling to keep up)\n  • Gap = 500 blocks → Q ≈ 0.00 (node is decoherent, possibly forked)\n\nAnalogy: Like clocks in a train station. High Q means all clocks show the same time. Low Q means one clock is minutes behind — passengers using that clock will miss their trains.`;
      }
    },
    { key: 'T', value: metrics.thermodynamic_efficiency, label: 'Thermodynamic Efficiency', exp: 0.20,
      tooltip: (v: number) => {
        const pct = (v * 100).toFixed(0);
        const weighted = Math.pow(v, 0.20);
        const grade = v > 0.9 ? 'Excellent — lean and efficient, minimal waste' : v > 0.7 ? 'Good — normal operating conditions' : v > 0.5 ? 'Warning — resource pressure building (check RAM/CPU)' : 'Critical — node is overloaded or starving for resources';
        return `T — Thermodynamic Efficiency: ${v.toFixed(4)} (${pct}%, weight 20%)\nWeighted contribution: ${v.toFixed(4)}^0.20 = ${weighted.toFixed(4)}\n\n${grade}\n\nWhat it measures: Is the node using its hardware resources efficiently?\n\nHow it's computed:\n  • T = (1 - cpu_waste) × (1 - mem_pressure) × io_efficiency\n  • cpu_waste = cpu_usage × (1 - blocks_produced / expected_blocks)\n  • mem_pressure = max(0, (rss_mb - comfortable_mb) / (total_mb - comfortable_mb))\n  • io_efficiency = 1.0 - (disk_wait_ms / 100ms)\n  • Swap usage is heavily penalized: any swap → T drops by 40%+\n\nResource thresholds:\n  • CPU <70% with steady blocks → T ≈ 0.95\n  • CPU 70-85% → T ≈ 0.80 (normal under load)\n  • CPU >90% with block stalls → T ≈ 0.30 (CPU-starved)\n  • RAM <80% of physical → no penalty\n  • RAM >90% or swapping → T drops to 0.10-0.30\n  • Disk latency <5ms → no penalty; >50ms → T halved\n\nReal-world example: Gamma (8GB RAM + 4GB swap) hits swap during sync → disk I/O 200ms+ → T = 0.08. This single factor drags K below 0.50, correctly blocking deployment to an overloaded node.`;
      }
    },
    { key: 'I', value: metrics.information_density, label: 'Information Density', exp: 0.15,
      tooltip: (v: number) => {
        const pct = (v * 100).toFixed(0);
        const weighted = Math.pow(v, 0.15);
        const grade = v > 0.9 ? 'Excellent — high signal, low noise' : v > 0.7 ? 'Good — healthy data throughput' : v > 0.5 ? 'Moderate — throughput below expected levels' : 'Low — node may be idle or processing mostly empty blocks';
        return `I — Information Density: ${v.toFixed(4)} (${pct}%, weight 15%)\nWeighted contribution: ${v.toFixed(4)}^0.15 = ${weighted.toFixed(4)}\n\n${grade}\n\nWhat it measures: How much useful data is this node processing per unit time?\n\nHow it's computed:\n  • I = (tx_throughput_ratio × 0.4) + (block_fill_rate × 0.3) + (peer_diversity × 0.3)\n  • tx_throughput_ratio = actual_tps / max_observed_tps (capped at 1.0)\n  • block_fill_rate = avg_txs_per_block / block_capacity\n  • peer_diversity = unique_peers_seen / total_expected_peers\n  • Normalizes for time-of-day: low traffic at 3 AM doesn't penalize\n\nScoring examples:\n  • 100 TPS, full blocks, 8 peers → I ≈ 0.95\n  • 50 TPS, half-full blocks, 4 peers → I ≈ 0.65\n  • 5 TPS, empty blocks, 2 peers → I ≈ 0.20\n  • 0 TPS (idle) → I ≈ 0.10 (baseline)\n\nWhy lowest weight (15%): Information density naturally fluctuates with network activity. Low traffic at 3 AM is normal, not alarming. This factor matters most when the network is busy: if peers see 200 TPS but this node only sees 50 TPS, I drops — correctly flagging a P2P message delivery problem. Shannon's information theory: high I = high signal-to-noise ratio.`;
      }
    },
    { key: 'R', value: metrics.network_resilience, label: 'Network Resilience', exp: 0.20,
      tooltip: (v: number) => {
        const pct = (v * 100).toFixed(0);
        const weighted = Math.pow(v, 0.20);
        const grade = v > 0.9 ? 'Excellent — battle-hardened, recovers instantly from disruptions' : v > 0.7 ? 'Good — stable connections, handles minor issues' : v > 0.5 ? 'Warning — connection instability or recent disconnections' : 'Critical — node is isolated or repeatedly losing peers';
        return `R — Network Resilience: ${v.toFixed(4)} (${pct}%, weight 20%)\nWeighted contribution: ${v.toFixed(4)}^0.20 = ${weighted.toFixed(4)}\n\n${grade}\n\nWhat it measures: Can the node maintain connections and recover from failures?\n\nHow it's computed:\n  • R = (peer_score × 0.30) + (uptime_score × 0.25) + (recovery_score × 0.25) + (delivery_score × 0.20)\n  • peer_score = connected_peers / expected_peers (e.g., 6/8 = 0.75)\n  • uptime_score = avg_connection_duration / target_duration (e.g., 4h/6h = 0.67)\n  • recovery_score = 1.0 - (avg_reconnect_time / 30s) (fast reconnect = high score)\n  • delivery_score = gossipsub_messages_delivered / messages_expected\n\nCascade effects (why 20% weight):\n  • R drops → node falls behind → Q (coherence) drops too\n  • R drops → missed blocks → T (efficiency) drops too\n  • This cascading amplification means network problems are self-revealing\n\nReal-world: Beta (Netherlands) loses transatlantic link to Delta (US) for 30s. R drops to 0.70 during the gap but recovers to 0.95 within seconds as Gamma provides an alternate path. A node stuck at R < 0.50 for >5 minutes likely has a firewall issue, not a transient outage.`;
      }
    },
  ];

  // v9.0.4: Color map per factor letter for consistent visual identity
  const factorColors: Record<string, { bar: string; text: string; bg: string }> = {
    'G': { bar: 'bg-emerald-500', text: 'text-emerald-400', bg: 'bg-emerald-500/20' },
    'Q': { bar: 'bg-cyan-500', text: 'text-cyan-400', bg: 'bg-cyan-500/20' },
    'T': { bar: 'bg-orange-500', text: 'text-orange-400', bg: 'bg-orange-500/20' },
    'I': { bar: 'bg-blue-500', text: 'text-blue-400', bg: 'bg-blue-500/20' },
    'R': { bar: 'bg-purple-500', text: 'text-purple-400', bg: 'bg-purple-500/20' },
  };

  return (
    <div className="flex items-center gap-1.5">
      {factors.map(f => {
        const fc = factorColors[f.key] || { bar: 'bg-emerald-500', text: 'text-amber-200/60', bg: 'bg-slate-700/30' };
        const barColor = f.value > 0.8 ? fc.bar : f.value > 0.5 ? 'bg-amber-500' : 'bg-red-500';
        return (
          <div key={f.key} className="flex-1 min-w-0">
            <KTooltip wide text={f.tooltip(f.value)}>
              <div className="cursor-help">
                <div className="flex items-center gap-1 mb-0.5">
                  <span className={`text-[9px] font-bold ${fc.text}`}>{f.key}</span>
                  <span className="text-[8px] text-amber-200/30 font-mono">{(f.value * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2.5 rounded bg-slate-700/40 overflow-hidden border border-slate-600/20">
                  <div className={`h-full rounded ${barColor} transition-all duration-700`}
                    style={{ width: `${Math.max(f.value * 100, 2)}%` }} />
                </div>
              </div>
            </KTooltip>
          </div>
        );
      })}
    </div>
  );
}

// Analytics tab: time-series history for sparkline charts
interface MetricsSnapshot {
  ts: number;
  // q-flux
  flux_rps: number;
  flux_err_pct: number;
  flux_active_conns: number;
  flux_upstream_active: number;
  flux_active_ws: number;
  flux_bytes_rx: number;
  flux_bytes_tx: number;
  flux_tls_fail_rate: number;
  flux_h2_active: number;
  // caddy
  caddy_rps: number;
  caddy_avg_ms: number;
  caddy_p99_ms: number;
  caddy_goroutines: number;
  caddy_memory_mb: number;
  caddy_err_pct: number;
  // network
  network_height: number;
  total_peers: number;
}

const MAX_HISTORY = 120; // 120 samples × 15s poll = 30 minutes

/** SVG Sparkline chart — renders a mini area chart from an array of numbers */
function Sparkline({ data, color, height = 32, width = 160, fill = true, label, format, unit }: {
  data: number[];
  color: string;
  height?: number;
  width?: number;
  fill?: boolean;
  label?: string;
  format?: (v: number) => string;
  unit?: string;
}) {
  if (data.length < 2) {
    return (
      <div style={{ width, height }} className="flex items-center justify-center text-[9px] text-slate-500">
        Collecting...
      </div>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 2;
  const chartH = height - pad * 2;
  const chartW = width - pad * 2;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const lineStr = points.join(' ');
  const areaStr = `${pad},${height - pad} ${lineStr} ${pad + chartW},${height - pad}`;

  const current = data[data.length - 1];
  const avg = data.reduce((a, b) => a + b, 0) / data.length;
  const fmt = format || ((v: number) => v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v % 1 === 0 ? String(v) : v.toFixed(1));

  return (
    <div>
      <svg width={width} height={height} className="block">
        {fill && (
          <polygon points={areaStr} fill={color} opacity={0.12} />
        )}
        <polyline points={lineStr} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
        {/* Current value dot */}
        {data.length > 0 && (() => {
          const lastX = pad + ((data.length - 1) / (data.length - 1)) * chartW;
          const lastY = pad + chartH - ((current - min) / range) * chartH;
          return <circle cx={lastX} cy={lastY} r="2.5" fill={color} />;
        })()}
      </svg>
      <div className="flex items-center justify-between mt-0.5">
        <span className="text-[8px] text-slate-500">min: {fmt(min)}{unit || ''}</span>
        <span className="text-[8px] text-slate-400">avg: {fmt(avg)}{unit || ''}</span>
        <span className="text-[8px] text-slate-500">max: {fmt(max)}{unit || ''}</span>
      </div>
    </div>
  );
}

/** Analytics metric card with sparkline */
function AnalyticsCard({ title, value, unit, data, color, tooltip, format }: {
  title: string;
  value: string;
  unit?: string;
  data: number[];
  color: string;
  tooltip?: string;
  format?: (v: number) => string;
}) {
  return (
    <div className="rounded-lg bg-slate-800/50 border border-slate-700/30 p-3 cursor-help" title={tooltip}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider">{title}</span>
        <div className="flex items-baseline gap-1">
          <span className="text-lg font-bold" style={{ color }}>{value}</span>
          {unit && <span className="text-[9px] text-slate-500">{unit}</span>}
        </div>
      </div>
      <Sparkline data={data} color={color} width={200} height={36} format={format} unit={unit ? ` ${unit}` : ''} />
    </div>
  );
}

function formatUptime(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  const hours = Math.floor(secs / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  if (hours < 24) return `${hours}h ${mins}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'ready':
      return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    case 'syncing':
      return <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />;
    case 'starting':
      return <Clock className="w-4 h-4 text-blue-400 animate-pulse" />;
    case 'offline':
      return <XCircle className="w-4 h-4 text-red-400" />;
    default:
      return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
  }
}

function formatEta(secs: number): string {
  if (secs <= 0) return '';
  if (secs < 60) return `~${secs}s`;
  if (secs < 3600) return `~${Math.floor(secs / 60)}m`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `~${h}h${m}m`;
}

// v1.0.2: Format hashrate with appropriate unit
function formatHashrate(hs: number): string {
  if (hs >= 1e12) return `${(hs / 1e12).toFixed(1)} TH/s`;
  if (hs >= 1e9) return `${(hs / 1e9).toFixed(1)} GH/s`;
  if (hs >= 1e6) return `${(hs / 1e6).toFixed(1)} MH/s`;
  if (hs >= 1e3) return `${(hs / 1e3).toFixed(1)} KH/s`;
  return `${hs} H/s`;
}

// v1.0.2: Mining Capacity Bar — queue health, hashrate, acceptance rate
function MiningCapacityBar({ cap }: { cap: MiningCapacityLocal }) {
  const queueColor = cap.queue_pct < 50 ? 'from-cyan-500 to-cyan-400'
    : cap.queue_pct < 80 ? 'from-amber-500 to-amber-400'
    : 'from-red-500 to-red-400';
  const queueTextColor = cap.queue_pct < 50 ? 'text-cyan-400'
    : cap.queue_pct < 80 ? 'text-amber-400'
    : 'text-red-400';

  const acceptColor = cap.acceptance_pct >= 95 ? 'from-emerald-500 to-emerald-400'
    : cap.acceptance_pct >= 80 ? 'from-amber-500 to-amber-400'
    : 'from-red-500 to-red-400';
  const acceptTextColor = cap.acceptance_pct >= 95 ? 'text-emerald-400'
    : cap.acceptance_pct >= 80 ? 'text-amber-400'
    : 'text-red-400';

  // Log-scale for hashrate bar (0 H/s = 0%, 1M H/s = 100%)
  const hrLog = cap.hashrate_hs > 0 ? Math.min(Math.log10(cap.hashrate_hs) / 6 * 100, 100) : 0;

  const isActive = cap.is_healthy && cap.last_solution_secs_ago < 300;
  const lastSolText = cap.last_solution_secs_ago < 60 ? `${cap.last_solution_secs_ago}s ago`
    : cap.last_solution_secs_ago < 3600 ? `${Math.floor(cap.last_solution_secs_ago / 60)}m ago`
    : cap.last_solution_secs_ago < 86400 ? `${Math.floor(cap.last_solution_secs_ago / 3600)}h ago`
    : cap.last_solution_secs_ago >= 4294967295 ? 'never' : `${Math.floor(cap.last_solution_secs_ago / 86400)}d ago`;

  return (
    <div className="mt-2 pt-2 border-t border-slate-700/30">
      {/* Mining status badge */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3 h-3 text-amber-200/50" />
          <span className="text-[10px] text-amber-200/50 font-medium">Mining</span>
        </div>
        <div className="flex items-center gap-1">
          <span className={`inline-block w-1.5 h-1.5 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className={`text-[9px] font-bold ${isActive ? 'text-emerald-300' : 'text-red-300'}`}>
            {isActive ? 'ACTIVE' : 'STALLED'}
          </span>
        </div>
      </div>

      {/* Queue health */}
      <div className="mb-1.5">
        <div className="flex justify-between text-[9px] mb-0.5">
          <span className="text-amber-200/40">Queue</span>
          <span className={queueTextColor}>{cap.queue_pct.toFixed(0)}% ({cap.shard_count} shards)</span>
        </div>
        <div className="w-full h-1 bg-slate-700/50 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${queueColor} transition-all duration-700`}
            style={{ width: `${Math.max(cap.queue_pct, 1)}%` }}
          />
        </div>
      </div>

      {/* Hashrate */}
      <div className="mb-1.5">
        <div className="flex justify-between text-[9px] mb-0.5">
          <span className="text-amber-200/40">Hashrate</span>
          <span className="text-cyan-300">{formatHashrate(cap.hashrate_hs)} ({cap.active_miners} miners)</span>
        </div>
        <div className="w-full h-1 bg-slate-700/50 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              cap.hashrate_hs > 0
                ? 'bg-gradient-to-r from-emerald-500 to-cyan-400 mining-shimmer'
                : 'bg-slate-600'
            }`}
            style={{ width: `${Math.max(hrLog, 1)}%` }}
          />
        </div>
      </div>

      {/* Acceptance rate */}
      <div className="mb-1">
        <div className="flex justify-between text-[9px] mb-0.5">
          <span className="text-amber-200/40">Accepted</span>
          <span className={acceptTextColor}>{cap.acceptance_pct.toFixed(1)}% ({cap.solutions_accepted.toLocaleString()}/{cap.solutions_submitted.toLocaleString()})</span>
        </div>
        <div className="w-full h-1 bg-slate-700/50 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${acceptColor} transition-all duration-700`}
            style={{ width: `${Math.max(cap.acceptance_pct, 1)}%` }}
          />
        </div>
      </div>

      {/* Last solution footer */}
      <div className="text-[8px] text-amber-200/30 text-right">
        Last solution: {lastSolText}
      </div>
    </div>
  );
}

function SyncModeBadge({ mode }: { mode: string }) {
  const config: Record<string, { label: string; color: string; bg: string }> = {
    fully_synced: { label: 'SYNCED', color: 'text-emerald-300', bg: 'bg-emerald-500/20' },
    turbo: { label: 'TURBO', color: 'text-blue-300', bg: 'bg-blue-500/20' },
    endgame: { label: 'ENDGAME', color: 'text-amber-300', bg: 'bg-amber-500/20' },
    micro: { label: 'MICRO', color: 'text-cyan-300', bg: 'bg-cyan-500/20' },
    idle: { label: 'IDLE', color: 'text-slate-400', bg: 'bg-slate-500/20' },
  };
  const c = config[mode] || config.idle;
  return (
    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${c.color} ${c.bg}`}>
      {c.label}
    </span>
  );
}

function ServerCard({ node, isActive, role, syncMetrics, miningCap }: { node: NodeStatus; isActive: boolean; role: 'canary' | 'primary' | 'backup' | 'bootstrap' | 'supernode'; syncMetrics?: SyncMetrics; miningCap?: MiningCapacityLocal | null }) {
  const [expanded, setExpanded] = useState(false);
  const sd = node.sync_details;
  const roleConfig = {
    canary: { label: 'CANARY', color: 'text-purple-300', bg: 'bg-purple-500/20', border: 'border-purple-400/40' },
    primary: { label: 'PRIMARY', color: 'text-emerald-300', bg: 'bg-emerald-500/20', border: 'border-emerald-400/40' },
    bootstrap: { label: 'BOOTSTRAP', color: 'text-cyan-300', bg: 'bg-cyan-500/20', border: 'border-cyan-400/40' },
    backup: { label: 'BACKUP', color: 'text-blue-300', bg: 'bg-blue-500/20', border: 'border-blue-400/40' },
    supernode: { label: 'SUPERNODE', color: 'text-amber-300', bg: 'bg-amber-500/20', border: 'border-amber-400/40' },
  }[role];

  // Prefer server-side speed/ETA when available
  const displaySpeed = sd && sd.blocks_per_second > 0 ? sd.blocks_per_second : (syncMetrics?.speed ?? 0);
  const displayEta = sd?.eta_seconds ?? (syncMetrics && syncMetrics.eta_secs > 0 ? syncMetrics.eta_secs : null);
  const isSyncing = sd ? sd.total_chunks > 0 && sd.completed_chunks < sd.total_chunks : false;

  return (
    <div className={`rounded-xl border p-4 relative ${
      node.online
        ? isActive
          ? 'border-emerald-400/50 bg-emerald-500/10 ring-1 ring-emerald-400/20'
          : 'border-emerald-500/30 bg-emerald-500/5'
        : 'border-red-500/30 bg-red-500/5'
    }`}>
      {/* Role + Active badges */}
      <div className="absolute -top-2 right-1 flex items-center gap-1">
        <div className={`flex items-center gap-0.5 px-1.5 py-0.5 rounded-full ${roleConfig.bg} border ${roleConfig.border}`}>
          <span className={`text-[9px] font-bold ${roleConfig.color}`}>{roleConfig.label}</span>
        </div>
        {isActive && node.online && (
          <div className="flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-400/40">
            <Radio className="w-2 h-2 text-emerald-400 animate-pulse" />
            <span className="text-[9px] font-bold text-emerald-300">ACTIVE</span>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Server className={`w-5 h-5 ${node.online ? 'text-emerald-400' : 'text-red-400'}`} />
          <span className="font-semibold text-amber-50">{node.name}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {sd && <SyncModeBadge mode={sd.sync_mode} />}
          <StatusIcon status={node.status} />
          <span className={`text-xs font-medium ${
            node.status === 'ready' ? 'text-emerald-400' :
            node.status === 'syncing' ? 'text-amber-400' :
            node.status === 'recovering' ? 'text-blue-400' :
            node.status === 'offline' ? 'text-red-400' :
            'text-blue-400'
          }`}>
            {node.online ? node.status.charAt(0).toUpperCase() + node.status.slice(1) : 'Offline'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center gap-1.5 text-amber-200/70">
          <Zap className="w-3 h-3" />
          <span>v{node.version || '?'}</span>
        </div>
        <div className="flex items-center gap-1.5 text-amber-200/70">
          <Layers className="w-3 h-3" />
          <span>{formatNumber(node.height)}</span>
          {(node as any)._recovering && (
            <span className="text-[9px] text-blue-400" title={`Verified: ${formatNumber((node as any)._verifiedHeight)}`}>
              (catching up)
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-amber-200/70">
          <Users className="w-3 h-3" />
          <span>{node.peers} peers</span>
        </div>
        <div className="flex items-center gap-1.5 text-amber-200/70">
          <Clock className="w-3 h-3" />
          <span>{formatUptime(node.uptime_secs)}</span>
        </div>
      </div>

      {/* Sync progress bar with speed and ETA */}
      {node.network_height > 0 && (
        <div className="mt-3">
          <div className="flex justify-between text-[10px] text-amber-200/50 mb-1">
            <span className="flex items-center gap-1">
              {displaySpeed > 0 ? (
                <><TrendingUp className="w-2.5 h-2.5 text-emerald-400" />{Math.round(displaySpeed).toLocaleString()} blk/s</>
              ) : (
                'Sync'
              )}
            </span>
            <span className="flex items-center gap-1.5">
              {displayEta != null && displayEta > 0 && (
                <span className="text-amber-300/70">ETA: {formatEta(displayEta)}</span>
              )}
              {(sd?.is_fully_synced || (syncMetrics && syncMetrics.eta_secs === -1)) && (
                <span className="text-emerald-400">synced</span>
              )}
              <span>{(syncMetrics?.sync_pct ?? (node.height / node.network_height) * 100).toFixed(1)}%</span>
            </span>
          </div>
          <div className="w-full h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000 ${
                (syncMetrics?.sync_pct ?? 0) >= 99.5
                  ? 'bg-emerald-500'
                  : 'bg-gradient-to-r from-amber-500 to-emerald-500'
              }`}
              style={{ width: `${Math.min((syncMetrics?.sync_pct ?? (node.height / node.network_height) * 100), 100)}%` }}
            />
          </div>
          {syncMetrics && syncMetrics.gap > 0 && (
            <div className="text-[9px] text-amber-200/40 mt-0.5 text-right">
              {syncMetrics.gap.toLocaleString()} blocks behind
            </div>
          )}
        </div>
      )}

      {/* Chunk progress (only when actively syncing) */}
      {sd && isSyncing && (
        <div className="mt-2">
          <div className="flex justify-between text-[10px] text-amber-200/50 mb-1">
            <span>Chunks: {sd.completed_chunks}/{sd.total_chunks} ({sd.chunk_progress_pct.toFixed(1)}%)</span>
            <span>In-flight: {sd.in_flight} | Queue: {sd.queued}</span>
          </div>
          <div className="w-full h-1 bg-slate-700/50 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500"
              style={{ width: `${Math.min(sd.chunk_progress_pct, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Mining capacity bars */}
      {miningCap && node.online && <MiningCapacityBar cap={miningCap} />}

      {/* Expandable details (click to toggle) */}
      {sd && node.online && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 w-full flex items-center justify-center gap-1 text-[9px] text-amber-200/40 hover:text-amber-200/70 transition-colors"
        >
          {expanded ? <ChevronDown className="w-2.5 h-2.5" /> : <ChevronRight className="w-2.5 h-2.5" />}
          {expanded ? 'Hide details' : 'Show details'}
        </button>
      )}
      {expanded && sd && (
        <div className="mt-1.5 pt-1.5 border-t border-slate-700/50 grid grid-cols-3 gap-x-3 gap-y-1 text-[9px] text-amber-200/50">
          <span>Speed: {sd.blocks_per_second > 0 ? `${Math.round(sd.blocks_per_second)} blk/s` : 'N/A'}</span>
          <span>DL: {sd.bytes_downloaded_mb > 0 ? `${sd.bytes_downloaded_mb.toFixed(1)} MB` : '0'}</span>
          <span>Ratio: {sd.compression_ratio > 0 ? `${sd.compression_ratio.toFixed(1)}x` : 'N/A'}</span>
          <span>Streams: {sd.active_streams}</span>
          <span>Failed: {sd.failed_chunks}</span>
          <span>Retried: {sd.retried_chunks}</span>
          <span>Peers: {sd.peer_count}</span>
          <span>Best: {sd.best_peer_height > 0 ? formatNumber(sd.best_peer_height) : 'N/A'}</span>
          <span>ETA: {sd.eta_seconds != null && sd.eta_seconds > 0 ? formatEta(sd.eta_seconds) : 'N/A'}</span>
          {/* v8.2.8: Apollo Subsystem Metrics */}
          {(sd.apollo_kalman_samples ?? 0) > 0 && (
            <>
              <span className="col-span-3 mt-1 text-cyan-300/70 font-bold border-t border-cyan-800/30 pt-1">APOLLO Kalman</span>
              <span>BW: {(sd.apollo_kalman_bandwidth_mbps ?? 0).toFixed(1)} Mbps</span>
              <span>Lat: {(sd.apollo_kalman_latency_ms ?? 0).toFixed(1)} ms</span>
              <span>Conf: {((sd.apollo_kalman_confidence ?? 0) * 100).toFixed(0)}%</span>
              <span>Chunk: {sd.apollo_kalman_optimal_chunk_kb ?? 0} KB</span>
              <span>Loss: {(sd.apollo_kalman_loss_pct ?? 0).toFixed(2)}%</span>
              <span>Jitter: {(sd.apollo_kalman_jitter_ms ?? 0).toFixed(1)} ms</span>
              <span>Timeout: {sd.apollo_kalman_timeout_ms ?? 0} ms</span>
              <span>Conc: {sd.apollo_kalman_concurrency ?? 0}</span>
              <span>Samples: {sd.apollo_kalman_samples ?? 0}</span>
            </>
          )}
          {(sd.apollo_pid_target_bps ?? 0) > 0 && (
            <>
              <span className="col-span-3 mt-1 text-green-300/70 font-bold border-t border-green-800/30 pt-1">APOLLO PID Controller</span>
              <span>Target: {Math.round(sd.apollo_pid_target_bps ?? 0)} bps</span>
              <span>Current: {(sd.apollo_pid_current_bps ?? 0).toFixed(1)} bps</span>
              <span>Error: {(sd.apollo_pid_error ?? 0).toFixed(3)}</span>
              <span>Kp: {(sd.apollo_pid_kp ?? 0).toFixed(3)}</span>
              <span>Ki: {(sd.apollo_pid_ki ?? 0).toFixed(3)}</span>
              <span>Kd: {(sd.apollo_pid_kd ?? 0).toFixed(3)}</span>
            </>
          )}
          {(sd.apollo_peers_tracked ?? 0) > 0 && (
            <>
              <span className="col-span-3 mt-1 text-purple-300/70 font-bold border-t border-purple-800/30 pt-1">APOLLO Gravity Assist</span>
              <span>Tracked: {sd.apollo_peers_tracked} peers</span>
              <span>Best: {sd.apollo_gravity_best_peer || 'N/A'}</span>
              <span>Heat: {(sd.apollo_gravity_best_heat ?? 0).toFixed(2)}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function DeployControlPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [deployStatus, setDeployStatus] = useState<DeployStatus | null>(null);
  const [convergence, setConvergence] = useState<ConvergenceStatus | null>(null);
  const [verifyEvents, setVerifyEvents] = useState<VerifyEvent[]>([]);
  const [isVerifying, setIsVerifying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [connInfo, setConnInfo] = useState(getConnectionInfo());
  const [sseStatus, setSseStatus] = useState<'connected' | 'reconnecting' | 'disconnected'>('disconnected');
  const [devFee, setDevFee] = useState<DevFeeStatus | null>(null);
  const [devFeeInput, setDevFeeInput] = useState('');
  const [devFeeSaving, setDevFeeSaving] = useState(false);
  const [devFeeMsg, setDevFeeMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [syncMetricsMap, setSyncMetricsMap] = useState<Record<string, SyncMetrics>>({});
  const [miningCapacity, setMiningCapacity] = useState<MiningCapacityAll | null>(null);
  const [caddyStats, setCaddyStats] = useState<CaddyStatsAll | null>(null);
  const [fluxStats, setFluxStats] = useState<FluxStats | null>(null);
  const [decentral, setDecentral] = useState<DecentralizationMetrics | null>(null);
  const prevHeightsRef = useRef<Record<string, { height: number; ts: number }>>({});
  // v8.2.9: Peak height tracking — never show a height decrease (prevents "rollback" scare)
  const peakHeightsRef = useRef<Record<string, number>>({});
  const eventSourceRef = useRef<EventSource | null>(null);
  const metricsHistoryRef = useRef<MetricsSnapshot[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<MetricsSnapshot[]>([]);

  // Tab state
  const [activeTab, setActiveTab] = useState<'overview' | 'analytics' | 'bank' | 'bridge' | 'settings' | 'bounty' | 'dex' | 'mining' | 'kparam' | 'compute'>('overview');

  // v9.1.4: Mining mode switch state
  const [miningModeStatus, setMiningModeStatus] = useState<{ forced_mode: string; pool_url: string | null } | null>(null);
  const [miningModeLoading, setMiningModeLoading] = useState(false);
  const [miningPoolUrlInput, setMiningPoolUrlInput] = useState('stratum+tcp://quillon.xyz:3333');
  const [miningModeMsg, setMiningModeMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Bounty admin state
  const [bountyStats, setBountyStats] = useState<any>(null);
  const [bountyBugs, setBountyBugs] = useState<any[]>([]);
  const [bountySocials, setBountySocials] = useState<any[]>([]);
  const [bountyLoading, setBountyLoading] = useState(false);
  const [bountyTab, setBountyTab] = useState<'bugs' | 'social'>('bugs');

  // v7.3.0: Node admin settings state
  const [isNodeAdmin, setIsNodeAdmin] = useState(false);
  const [adminSettings, setAdminSettings] = useState<any>(null);
  const [oauthConsents, setOauthConsents] = useState<any[]>([]);
  const [nodeInfo, setNodeInfo] = useState<any>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);

  // Bridge state (v7.2.5)
  const [bridgeData, setBridgeData] = useState<any>(null);

  // Bank CLI state
  const [bankMetrics, setBankMetrics] = useState<any>(null);
  const [bankLoans, setBankLoans] = useState<any[]>([]);
  const [bankAtRisk, setBankAtRisk] = useState<any[]>([]);
  const [bankReserves, setBankReserves] = useState<any>(null);
  const [bankMessages, setBankMessages] = useState<any[]>([]);
  const [bankLoading, setBankLoading] = useState(false);
  const [bankSection, setBankSection] = useState<string>('dashboard');
  const [bankCmdLog, setBankCmdLog] = useState<Array<{ cmd: string; result: string; ok: boolean; ts: number }>>([]);
  const [mintAmount, setMintAmount] = useState('');
  const [mintCollateral, setMintCollateral] = useState('');
  const [mintWallet, setMintWallet] = useState('');
  const [selectedLoanId, setSelectedLoanId] = useState('');
  const [respondMsgId, setRespondMsgId] = useState('');
  const [respondContent, setRespondContent] = useState('');
  const bankLogRef = useRef<HTMLDivElement>(null);

  // v8.5.3: DEX Analytics state
  const [dexPools, setDexPools] = useState<any[]>([]);
  const [dexFeeStats, setDexFeeStats] = useState<any>(null);
  const [dexLoading, setDexLoading] = useState(false);
  const [dexSection, setDexSection] = useState<'analytics' | 'pools' | 'fees'>('analytics');

  // v9.3.1: K-Parameter Gauge state
  const [kParamData, setKParamData] = useState<any>(null);
  const [kParamLoading, setKParamLoading] = useState(false);
  const [kParamError, setKParamError] = useState<string | null>(null);
  const [kParamHistory, setKParamHistory] = useState<Array<{ k: number; phase: string; ts: number }>>([]);
  const [showOrbitalInfo, setShowOrbitalInfo] = useState(false);

  // v9.5.0: Starship Endgame — Compute orchestrator state
  const [computeData, setComputeData] = useState<any>(null);
  const [computeLoading, setComputeLoading] = useState(false);

  // Check if current wallet is master
  const walletAddress = localStorage.getItem('walletAddress') || '';
  const cleanWallet = walletAddress.replace('qnk', '').replace('qug', '');
  const isMasterWallet = cleanWallet === MASTER_WALLET;
  const isMaster = isMasterWallet || isNodeAdmin; // Node admin can also open the panel
  const isLoggedIn = !!walletAddress; // v8.6.4: Any logged-in user can see status

  // v7.3.0: Check if current wallet is the node's --admin-wallet
  useEffect(() => {
    if (!walletAddress) return;
    fetch('/api/v1/admin/is-admin', {
      headers: { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}` },
    })
      .then(r => r.json())
      .then(data => setIsNodeAdmin(data.is_admin === true))
      .catch(() => setIsNodeAdmin(false));
  }, [walletAddress]);

  // Listen for open event from TopBar admin button
  useEffect(() => {
    const handler = () => {
      // If node admin but not founder, default to settings tab
      if (isNodeAdmin && !isMasterWallet) {
        setActiveTab('settings');
      }
      setIsOpen(true);
    };
    window.addEventListener('open-deploy-panel', handler);
    return () => window.removeEventListener('open-deploy-panel', handler);
  }, [isNodeAdmin, isMasterWallet]);

  // Track connection info changes (failover events)
  useEffect(() => {
    const updateConnInfo = () => setConnInfo(getConnectionInfo());
    window.addEventListener('api-failover', updateConnInfo);
    // Also poll periodically in case of subtle changes
    const interval = setInterval(updateConnInfo, 5000);
    return () => {
      window.removeEventListener('api-failover', updateConnInfo);
      clearInterval(interval);
    };
  }, []);

  // Track SSE connection status
  useEffect(() => {
    const handleSseConnected = () => setSseStatus('connected');
    const handleSseDisconnected = () => setSseStatus('disconnected');
    const handleSseReconnecting = () => setSseStatus('reconnecting');

    // Listen for SSE events dispatched by App.tsx
    window.addEventListener('sse-connected', handleSseConnected);
    window.addEventListener('sse-disconnected', handleSseDisconnected);
    window.addEventListener('sse-reconnecting', handleSseReconnecting);

    // Check if we have an active SSE by looking for recent block events
    const checkSse = () => {
      const lastBlock = localStorage.getItem('lastBlockTime');
      if (lastBlock) {
        const elapsed = Date.now() - parseInt(lastBlock);
        setSseStatus(elapsed < 30000 ? 'connected' : 'reconnecting');
      }
    };
    checkSse();
    const interval = setInterval(checkSse, 5000);

    return () => {
      window.removeEventListener('sse-connected', handleSseConnected);
      window.removeEventListener('sse-disconnected', handleSseDisconnected);
      window.removeEventListener('sse-reconnecting', handleSseReconnecting);
      clearInterval(interval);
    };
  }, []);

  // Fetch deploy status + convergence data in parallel
  // v8.6.4: Status is public, convergence/dev-fee may require admin
  const fetchStatus = useCallback(async () => {
    if (!isLoggedIn) return;
    setLoading(true);
    setError(null);
    const headers = {
      'X-Wallet-Auth': walletAddress,
      'Authorization': `Bearer ${walletAddress}`,
    };
    try {
      const [statusResp, convResp, devFeeResp, mCapResp, diResp] = await Promise.all([
        fetch('/api/v1/admin/deploy/status', { headers }),
        fetch('/api/v1/admin/deploy/convergence', { headers }).catch(() => null),
        isMaster ? fetch('/api/v1/admin/dev-fee', { headers }).catch(() => null) : Promise.resolve(null),
        isMaster ? fetch('/api/v1/admin/mining/capacity', { headers }).catch(() => null) : Promise.resolve(null),
        isMaster ? fetch('/api/v1/admin/decentralization', { headers }).catch(() => null) : Promise.resolve(null),
      ]);

      if (statusResp.status === 403) {
        setError('Access denied - not master wallet');
        return;
      }

      // Parse status
      const text = await statusResp.text();
      if (text) {
        const json = JSON.parse(text);
        if (json.data) {
          const data = json.data;
          if (!data.alpha) {
            data.alpha = {
              name: 'Server Alpha', url: 'http://161.35.219.10:8080',
              online: false, version: '', height: 0, network_height: 0,
              peers: 0, uptime_secs: 0, status: 'offline',
            };
          }
          if (!data.delta) {
            data.delta = {
              name: 'Server Delta', url: 'http://5.79.79.158:8080',
              online: false, version: '', height: 0, network_height: 0,
              peers: 0, uptime_secs: 0, status: 'offline',
            };
          }
          if (!data.epsilon) {
            data.epsilon = {
              name: 'Server Epsilon', url: 'http://89.149.241.126:8080',
              online: false, version: '', height: 0, network_height: 0,
              peers: 0, uptime_secs: 0, status: 'offline',
            };
          }
          // v8.2.9: Enforce peak heights — NEVER show a height decrease
          // This prevents the "rollback scare" when a node restarts and syncs back up
          for (const [key, node] of Object.entries({ alpha: data.alpha, beta: data.beta, gamma: data.gamma, delta: data.delta, epsilon: data.epsilon }) as [string, any][]) {
            if (!node || !node.online || node.height === 0) continue;
            const prevPeak = peakHeightsRef.current[key] || 0;
            if (node.height > prevPeak) {
              peakHeightsRef.current[key] = node.height;
            } else if (node.height < prevPeak && prevPeak > 0) {
              // Node restarted and is catching up — show peak height and "recovering" status
              node._recovering = true;
              node._peakHeight = prevPeak;
              node._verifiedHeight = node.height;
              // Show the peak height so users never see a decrease
              node.height = prevPeak;
              // Override status to "recovering" so it's clear what's happening
              if (node.status === 'syncing' || node.status === 'ready') {
                node.status = 'recovering';
              }
            }
          }
          setDeployStatus(data);
          setLastRefresh(new Date());
        }
      }

      // Parse convergence (graceful — old backend may not have this endpoint)
      if (convResp && convResp.ok) {
        try {
          const convJson = await convResp.json();
          if (convJson.data) {
            setConvergence(convJson.data);
          }
        } catch {}
      }

      // Parse dev fee status
      if (devFeeResp && devFeeResp.ok) {
        try {
          const devFeeJson = await devFeeResp.json();
          if (devFeeJson.data) {
            setDevFee(devFeeJson.data);
            if (!devFeeInput) {
              setDevFeeInput(String(devFeeJson.data.fee_bps));
            }
          }
        } catch {}
      }

      // Parse mining capacity
      if (mCapResp && mCapResp.ok) {
        try {
          const mCapJson = await mCapResp.json();
          if (mCapJson.data) {
            setMiningCapacity(mCapJson.data);
          }
        } catch {}
      }

      // Parse decentralization index
      if (diResp && diResp.ok) {
        try {
          const diJson = await diResp.json();
          if (diJson.data) setDecentral(diJson.data);
        } catch {}
      }

      // v8.9.9: Fetch caddy + flux stats (admin only)
      let latestCaddy: CaddyStatsAll | null = null;
      let latestFlux: FluxStats | null = null;
      if (isMaster) {
        try {
          const caddyResp = await fetch('/api/v1/admin/caddy/stats', {
            headers: { 'X-Wallet-Auth': walletAddress || '' },
          });
          if (caddyResp.ok) {
            const caddyJson = await caddyResp.json();
            if (caddyJson.data) {
              latestCaddy = caddyJson.data;
              setCaddyStats(caddyJson.data);
            }
          }
        } catch {}
        // v9.2.0: Fetch q-flux stats
        try {
          const fluxResp = await fetch('/api/v1/admin/flux/stats', {
            headers: { 'X-Wallet-Auth': walletAddress || '' },
          });
          if (fluxResp.ok) {
            const fluxJson = await fluxResp.json();
            if (fluxJson.data) {
              latestFlux = fluxJson.data;
              setFluxStats(fluxJson.data);
            }
          }
        } catch {}
      }

      // v9.2.1: Collect analytics history snapshot (reuse already-fetched data)
      {
        const snap: MetricsSnapshot = {
          ts: Date.now(),
          flux_rps: 0, flux_err_pct: 0, flux_active_conns: 0, flux_upstream_active: 0,
          flux_active_ws: 0, flux_bytes_rx: 0, flux_bytes_tx: 0, flux_tls_fail_rate: 0, flux_h2_active: 0,
          caddy_rps: 0, caddy_avg_ms: 0, caddy_p99_ms: 0, caddy_goroutines: 0, caddy_memory_mb: 0, caddy_err_pct: 0,
          network_height: 0, total_peers: 0,
        };
        if (latestFlux) {
          const fx = latestFlux;
          snap.flux_rps = fx.requests_per_second || 0;
          snap.flux_err_pct = fx.error_rate_pct || 0;
          snap.flux_active_conns = fx.active_connections || 0;
          snap.flux_upstream_active = fx.upstream_active || 0;
          snap.flux_active_ws = fx.active_websockets || 0;
          snap.flux_bytes_rx = fx.bytes_received || 0;
          snap.flux_bytes_tx = fx.bytes_sent || 0;
          const totalTls = (fx.tls_handshakes || 0) + (fx.tls_handshake_failures || 0);
          snap.flux_tls_fail_rate = totalTls > 0 ? ((fx.tls_handshake_failures || 0) / totalTls) * 100 : 0;
          snap.flux_h2_active = (fx.h2_streams_opened || 0) - (fx.h2_streams_closed || 0);
        }
        const ce = (latestCaddy as any)?.epsilon;
        if (ce) {
          snap.caddy_rps = ce.requests_per_second || 0;
          snap.caddy_avg_ms = ce.avg_response_ms || 0;
          snap.caddy_p99_ms = ce.p99_response_ms || 0;
          snap.caddy_goroutines = ce.goroutines || 0;
          snap.caddy_memory_mb = ce.memory_mb || 0;
          const tot = (ce.requests_by_status?.ok_2xx || 0) + (ce.requests_by_status?.client_err_4xx || 0) + (ce.requests_by_status?.server_err_5xx || 0);
          snap.caddy_err_pct = tot > 0 ? ((ce.requests_by_status?.server_err_5xx || 0) / tot) * 100 : 0;
        }

        const hist = metricsHistoryRef.current;
        hist.push(snap);
        if (hist.length > MAX_HISTORY) hist.splice(0, hist.length - MAX_HISTORY);
        metricsHistoryRef.current = hist;
        setMetricsHistory([...hist]);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, [isMaster, walletAddress]);

  // Bank API helper
  const bankHeaders = {
    'X-Wallet-Auth': walletAddress,
    'Authorization': `Bearer ${walletAddress}`,
    'Content-Type': 'application/json',
  };

  const addBankLog = useCallback((cmd: string, result: string, ok: boolean) => {
    setBankCmdLog(prev => [...prev.slice(-49), { cmd, result, ok, ts: Date.now() }]);
    setTimeout(() => bankLogRef.current?.scrollTo({ top: bankLogRef.current.scrollHeight, behavior: 'smooth' }), 100);
  }, []);

  const fetchBankData = useCallback(async () => {
    if (!isMaster) return;
    setBankLoading(true);
    try {
      const [metricsR, loansR, riskR, reservesR, msgsR] = await Promise.all([
        fetch('/api/v1/quillon-bank/metrics', { headers: bankHeaders }).catch(() => null),
        fetch('/api/v1/quillon-bank/lending/applications', { headers: bankHeaders }).catch(() => null),
        fetch('/api/v1/quillon-bank/lending/at-risk', { headers: bankHeaders }).catch(() => null),
        fetch('/api/v1/quillon-bank/treasury/reserves', { headers: bankHeaders }).catch(() => null),
        fetch('/api/v1/quillon-bank/messages/admin/list', { headers: bankHeaders }).catch(() => null),
      ]);

      if (metricsR?.ok) {
        try { const j = await metricsR.json(); setBankMetrics(j.data || j); } catch {}
      }
      if (loansR?.ok) {
        try { const j = await loansR.json(); const d = j.data || j; setBankLoans(Array.isArray(d) ? d : []); } catch {}
      }
      if (riskR?.ok) {
        try { const j = await riskR.json(); const d = j.data || j; setBankAtRisk(Array.isArray(d) ? d : []); } catch {}
      }
      if (reservesR?.ok) {
        try { const j = await reservesR.json(); setBankReserves(j.data || j); } catch {}
      }
      if (msgsR?.ok) {
        try { const j = await msgsR.json(); const d = j.data || j; setBankMessages(Array.isArray(d) ? d : []); } catch {}
      }
    } catch {}
    setBankLoading(false);
  }, [isMaster, walletAddress]);

  // v8.5.3: Fetch DEX analytics data
  const fetchDexData = useCallback(async () => {
    if (!isMaster) return;
    setDexLoading(true);
    try {
      const [poolsR, feesR, supplyR] = await Promise.all([
        fetch('/api/v1/dex/pools').catch(() => null),
        fetch('/api/v1/admin/operator-fees', { headers: { 'x-wallet-address': walletAddress } }).catch(() => null),
        fetch('/api/v1/network/supply').catch(() => null),
      ]);

      if (poolsR?.ok) {
        try {
          const j = await poolsR.json();
          const pools = j.data || j.pools || [];
          setDexPools(Array.isArray(pools) ? pools : []);
        } catch {}
      }
      if (feesR?.ok) {
        try { const j = await feesR.json(); setDexFeeStats(j.data || j); } catch {}
      }
    } catch {}
    setDexLoading(false);
  }, [isMaster, walletAddress]);

  // v9.1.4: Mining mode switch
  const fetchMiningModeStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/v1/admin/mining/mode-status', {
        headers: { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}` },
      });
      if (resp.ok) {
        const json = await resp.json();
        setMiningModeStatus(json);
      }
    } catch {}
  }, [walletAddress]);

  // v9.3.1: Fetch K-parameter gauge data
  const fetchKParamData = useCallback(async () => {
    setKParamLoading(true);
    setKParamError(null);
    try {
      const resp = await fetch('/api/v1/k-parameter');
      if (resp.ok) {
        const json = await resp.json();
        const d = json.data || json;
        setKParamData(d);
        setKParamError(null);
        // Append to history (keep last 30 data points = 30 minutes)
        setKParamHistory(prev => {
          const next = [...prev, { k: d.k_value ?? 0, phase: d.phase ?? 'stable', ts: Date.now() }];
          return next.slice(-30);
        });
      } else {
        setKParamError(`Server returned ${resp.status} — endpoint may not be deployed yet`);
      }
    } catch (e: any) {
      setKParamError(e?.message || 'Failed to reach server');
    }
    setKParamLoading(false);
  }, []);

  // v9.5.0: Fetch compute orchestrator status
  const fetchComputeData = useCallback(async () => {
    setComputeLoading(true);
    try {
      const resp = await fetch('/api/v1/compute/status');
      if (resp.ok) {
        const json = await resp.json();
        setComputeData(json.data || json);
      }
    } catch (_) { /* ignore */ }
    setComputeLoading(false);
  }, []);

  const triggerMiningModeSwitch = useCallback(async (targetMode: string) => {
    setMiningModeLoading(true);
    setMiningModeMsg(null);
    try {
      const body: any = { target_mode: targetMode };
      if (targetMode === 'pool') {
        body.pool_url = miningPoolUrlInput;
      }
      body.reason = 'Admin panel switch';

      const resp = await fetch('/api/v1/admin/mining/mode-switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': walletAddress,
          'Authorization': `Bearer ${walletAddress}`,
        },
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        const json = await resp.json();
        setMiningModeMsg({ type: 'success', text: `Switched to ${json.new_mode} mode (${json.sse_subscribers} SSE subscribers notified)` });
        fetchMiningModeStatus();
      } else {
        setMiningModeMsg({ type: 'error', text: `Failed: HTTP ${resp.status}` });
      }
    } catch (e: any) {
      setMiningModeMsg({ type: 'error', text: e.message || 'Unknown error' });
    }
    setMiningModeLoading(false);
  }, [walletAddress, miningPoolUrlInput, fetchMiningModeStatus]);

  // Bank admin actions
  const bankAction = useCallback(async (cmd: string, method: string, path: string, body?: any) => {
    addBankLog(cmd, 'Executing...', true);
    try {
      const resp = await fetch(path, {
        method,
        headers: bankHeaders,
        body: body ? JSON.stringify(body) : undefined,
      });
      const text = await resp.text();
      let parsed: any;
      try { parsed = JSON.parse(text); } catch { parsed = text; }
      if (resp.ok) {
        addBankLog(cmd, typeof parsed === 'string' ? parsed : JSON.stringify(parsed.data || parsed, null, 2), true);
        fetchBankData();
      } else {
        addBankLog(cmd, `Error ${resp.status}: ${typeof parsed === 'string' ? parsed : JSON.stringify(parsed)}`, false);
      }
    } catch (e: any) {
      addBankLog(cmd, `Network error: ${e.message}`, false);
    }
  }, [walletAddress, addBankLog, fetchBankData]);

  // Auto-refresh when panel opens
  // v8.6.4: Status refresh for all logged-in users, bank/bridge only for isMaster
  useEffect(() => {
    if (isOpen && isLoggedIn) {
      fetchStatus();
      setConnInfo(getConnectionInfo());
      if (isMaster && activeTab === 'bank') fetchBankData();
      if (isMaster && activeTab === 'bridge') {
        fetch('/api/v1/bridge/status').then(r => r.json()).then(d => { if (d.success) setBridgeData(d.data); }).catch(() => {});
      }
      if (activeTab === 'kparam') fetchKParamData();
      const interval = setInterval(() => {
        fetchStatus();
        setConnInfo(getConnectionInfo());
        if (isMaster && activeTab === 'bank') fetchBankData();
        if (isMaster && activeTab === 'bridge') {
          fetch('/api/v1/bridge/status').then(r => r.json()).then(d => { if (d.success) setBridgeData(d.data); }).catch(() => {});
        }
        if (activeTab === 'kparam') fetchKParamData();
      }, 15000);
      return () => clearInterval(interval);
    }
  }, [isOpen, isLoggedIn, isMaster, fetchStatus, activeTab, fetchBankData, fetchKParamData]);

  // v7.3.0: Fetch node settings data when settings tab is active
  const fetchSettingsData = useCallback(async () => {
    const hdrs = { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}` };
    setSettingsLoading(true);
    try {
      const [settingsRes, consentsRes, nodeRes] = await Promise.all([
        fetch('/api/v1/admin/settings', { headers: hdrs }).catch(() => null),
        fetch('/api/v1/admin/oauth2/consents', { headers: hdrs }).catch(() => null),
        fetch('/api/v1/admin/node/info', { headers: hdrs }).catch(() => null),
      ]);
      if (settingsRes?.ok) setAdminSettings(await settingsRes.json());
      if (consentsRes?.ok) setOauthConsents(await consentsRes.json());
      if (nodeRes?.ok) setNodeInfo(await nodeRes.json());
    } catch { /* ignore */ }
    setSettingsLoading(false);
  }, [walletAddress]);

  useEffect(() => {
    if (isOpen && isNodeAdmin && activeTab === 'settings') {
      fetchSettingsData();
    }
  }, [isOpen, isNodeAdmin, activeTab, fetchSettingsData]);

  // Fetch bounty admin data
  const BOUNTY_API = '/bounty-api';
  const bountyAdminHeaders = { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}` };
  const fetchBountyData = useCallback(async () => {
    setBountyLoading(true);
    try {
      const hdrs = { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}` };
      const [statsRes, bugsRes, socialsRes] = await Promise.all([
        fetch(`${BOUNTY_API}/v1/admin/stats`, { headers: hdrs }).catch(() => null),
        fetch(`${BOUNTY_API}/v1/admin/bug-reports`, { headers: hdrs }).catch(() => null),
        fetch(`${BOUNTY_API}/v1/admin/social-activities`, { headers: hdrs }).catch(() => null),
      ]);
      if (statsRes?.ok) setBountyStats(await statsRes.json());
      if (bugsRes?.ok) setBountyBugs(await bugsRes.json());
      if (socialsRes?.ok) setBountySocials(await socialsRes.json());
    } catch { /* ignore */ }
    setBountyLoading(false);
  }, [walletAddress]);

  useEffect(() => {
    if (isOpen && isMasterWallet && activeTab === 'bounty') {
      fetchBountyData();
    }
  }, [isOpen, isMasterWallet, activeTab, fetchBountyData]);

  const updateBugStatus = async (userId: string, timestamp: number, status: string) => {
    try {
      const res = await fetch(`${BOUNTY_API}/v1/admin/bug-report/update`, {
        method: 'POST',
        headers: { ...bountyAdminHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, timestamp, status }),
      });
      if (res.ok) fetchBountyData();
    } catch { /* ignore */ }
  };

  const updateSocialStatus = async (userId: string, platform: number, timestamp: number, verified: boolean) => {
    try {
      const res = await fetch(`${BOUNTY_API}/v1/admin/social-activity/update`, {
        method: 'POST',
        headers: { ...bountyAdminHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, platform, timestamp, verified }),
      });
      if (res.ok) fetchBountyData();
    } catch { /* ignore */ }
  };

  // Compute sync metrics (speed, ETA) whenever deployStatus changes
  useEffect(() => {
    if (!deployStatus) return;
    const now = Date.now();
    const newMetrics: Record<string, SyncMetrics> = {};

    for (const [key, node] of Object.entries({
      alpha: deployStatus.alpha,
      beta: deployStatus.beta,
      gamma: deployStatus.gamma,
      delta: deployStatus.delta,
    })) {
      if (!node || !node.online) continue;

      const prev = prevHeightsRef.current[key];
      const netH = node.network_height || 0;
      const gap = netH > node.height ? netH - node.height : 0;
      const syncPct = netH > 0 ? Math.min((node.height / netH) * 100, 100) : 0;

      let speed = 0;
      let etaSecs = 0;

      if (prev && prev.height > 0 && node.height > prev.height) {
        const elapsed = (now - prev.ts) / 1000;
        if (elapsed > 0) {
          speed = Math.round((node.height - prev.height) / elapsed);
          if (speed > 0 && gap > 0) {
            etaSecs = Math.round(gap / speed);
          }
        }
      }

      newMetrics[key] = {
        speed,
        eta_secs: gap <= 50 ? -1 : etaSecs,
        sync_pct: syncPct,
        gap,
      };

      // Update previous heights for next calculation
      prevHeightsRef.current[key] = { height: node.height, ts: now };
    }

    setSyncMetricsMap(newMetrics);
  }, [deployStatus]);

  // Start verification
  const startVerification = async () => {
    setIsVerifying(true);
    setVerifyEvents([]);
    setError(null);

    try {
      await fetch('/api/v1/admin/deploy/verify', {
        method: 'POST',
        headers: {
          'X-Wallet-Auth': walletAddress,
          'Authorization': `Bearer ${walletAddress}`,
        },
      });

      const baseUrl = window.location.origin;
      const url = `${baseUrl}/api/v1/admin/deploy/progress`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.addEventListener('verify-progress', (e) => {
        try {
          const event: VerifyEvent = JSON.parse(e.data);
          setVerifyEvents(prev => [...prev, event]);
        } catch {}
      });

      es.addEventListener('verify-complete', () => {
        setIsVerifying(false);
        es.close();
        eventSourceRef.current = null;
        fetchStatus();
      });

      es.onerror = () => {
        setIsVerifying(false);
        es.close();
        eventSourceRef.current = null;
      };
    } catch (e: any) {
      setError(e.message || 'Failed to start verification');
      setIsVerifying(false);
    }
  };

  // v5.6.0: Stream pipeline progress via SSE (no auth needed on this endpoint)
  const startPipelineStream = useCallback(() => {
    const baseUrl = window.location.origin;
    const url = `${baseUrl}/api/v1/admin/deploy/progress`;
    const es = new EventSource(url);
    eventSourceRef.current = es;
    setPipelineRunning(true);

    es.addEventListener('verify-progress', (e) => {
      try {
        const event: VerifyEvent = JSON.parse(e.data);
        setVerifyEvents(prev => [...prev, event]);
      } catch {}
    });

    es.addEventListener('verify-complete', () => {
      setPipelineRunning(false);
      es.close();
      eventSourceRef.current = null;
      fetchStatus();
    });

    es.onerror = () => {
      setPipelineRunning(false);
      es.close();
      eventSourceRef.current = null;
    };
  }, [fetchStatus]);

  // v5.6.0: Trigger full deploy pipeline
  const triggerDeployAll = useCallback(async () => {
    if (!confirm('Deploy to all 5 servers?\n\nPipeline: Epsilon+Alpha+Delta (parallel) -> Gamma (verify) -> Beta (primary)')) return;
    setError(null);
    setVerifyEvents([]);
    try {
      const resp = await fetch('/api/v1/admin/deploy/promote', {
        method: 'POST',
        headers: {
          'X-Wallet-Auth': walletAddress,
          'Authorization': `Bearer ${walletAddress}`,
        },
      });
      if (!resp.ok) {
        setError(`Deploy trigger failed: ${resp.status}`);
        return;
      }
      // Start streaming progress
      startPipelineStream();
    } catch (e: any) {
      setError('Failed to trigger deploy: ' + (e.message || 'Unknown error'));
    }
  }, [walletAddress, startPipelineStream]);

  // v5.6.0: Trigger rollback
  const triggerRollback = useCallback(async () => {
    if (!confirm('Rollback to previous binary on all servers?')) return;
    setError(null);
    setVerifyEvents([]);
    try {
      const resp = await fetch('/api/v1/admin/deploy/rollback', {
        method: 'POST',
        headers: {
          'X-Wallet-Auth': walletAddress,
          'Authorization': `Bearer ${walletAddress}`,
        },
      });
      if (!resp.ok) {
        setError(`Rollback trigger failed: ${resp.status}`);
        return;
      }
      startPipelineStream();
    } catch (e: any) {
      setError('Failed to trigger rollback: ' + (e.message || 'Unknown error'));
    }
  }, [walletAddress, startPipelineStream]);

  // Save dev fee config
  const saveDevFee = useCallback(async () => {
    const bps = parseInt(devFeeInput);
    if (isNaN(bps) || bps < 0 || bps > 1000) {
      setDevFeeMsg({ type: 'error', text: 'Fee must be 0-1000 bps (0%-10%)' });
      return;
    }
    setDevFeeSaving(true);
    setDevFeeMsg(null);
    try {
      const resp = await fetch('/api/v1/admin/dev-fee/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': walletAddress,
          'Authorization': `Bearer ${walletAddress}`,
        },
        body: JSON.stringify({ fee_bps: bps }),
      });
      if (!resp.ok) {
        setDevFeeMsg({ type: 'error', text: `Failed: HTTP ${resp.status}` });
        return;
      }
      const json = await resp.json();
      if (json.data) {
        setDevFee(json.data);
        setDevFeeMsg({ type: 'success', text: `Updated to ${bps} bps (${(bps / 100).toFixed(2)}%)` });
        setTimeout(() => setDevFeeMsg(null), 3000);
      } else if (json.error) {
        setDevFeeMsg({ type: 'error', text: json.error });
      }
    } catch (e: any) {
      setDevFeeMsg({ type: 'error', text: e.message || 'Failed to save' });
    } finally {
      setDevFeeSaving(false);
    }
  }, [devFeeInput, walletAddress]);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // v8.6.4: Show status for all logged-in users; admin features need isMaster
  if (!isLoggedIn) return null;

  const allPassed = verifyEvents.length > 0 &&
    verifyEvents.some(e => e.step === 'RESULT' && e.status === 'passed');
  const anyFailed = verifyEvents.some(e => e.status === 'failed');

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Mining shimmer animation */}
          <style>{`
            @keyframes mining-shimmer {
              0% { background-position: -200% center; }
              100% { background-position: 200% center; }
            }
            .mining-shimmer {
              background-size: 200% 100%;
              animation: mining-shimmer 2s linear infinite;
            }
          `}</style>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[99998]"
            onClick={() => setIsOpen(false)}
          />

          {/* Panel — pinned to top of screen with large z-index */}
          <motion.div
            initial={{ opacity: 0, y: -40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed top-4 left-1/2 transform -translate-x-1/2 w-[1640px] max-w-[95vw] max-h-[90vh] overflow-y-auto rounded-2xl z-[99999]"
            style={{
              background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.98) 0%, rgba(30, 20, 55, 0.98) 100%)',
              border: '2px solid rgba(16, 185, 129, 0.3)',
              boxShadow: '0 25px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(16, 185, 129, 0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-5 pb-3 border-b border-emerald-500/20">
              <div className="flex items-center gap-3">
                <Shield className="w-6 h-6 text-emerald-400" />
                <div>
                  <h2 className="text-lg font-bold text-emerald-50">Node Admin</h2>
                  <p className="text-xs text-emerald-300/60">
                    {activeTab === 'overview' ? 'Deploy Control Panel' : activeTab === 'analytics' ? 'Live Analytics' : activeTab === 'settings' ? 'Node Settings' : activeTab === 'bridge' ? 'Bridge Pairs' : activeTab === 'bounty' ? 'Bounty Campaign Admin' : activeTab === 'dex' ? 'DEX Analytics' : activeTab === 'mining' ? 'Mining Mode Control' : activeTab === 'kparam' ? 'K-Parameter Health Gauge' : activeTab === 'compute' ? 'Starship Compute' : 'Quillon Bank CLI'}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-5 h-5 text-emerald-300/60" />
              </button>
            </div>

            {/* Tab Bar */}
            <div className="flex items-center gap-1 px-5 pt-3 pb-0">
              {([
                // v8.6.4: Servers tab visible for all logged-in users
                { id: 'overview' as const, icon: Server, label: 'Servers' },
                { id: 'kparam' as const, icon: Activity, label: 'K-Param' },
                ...(isMasterWallet ? [
                  { id: 'analytics' as const, icon: BarChart3, label: 'Analytics' },
                  { id: 'bank' as const, icon: Landmark, label: 'Bank CLI' },
                  { id: 'bridge' as const, icon: Globe, label: 'Bridge Pairs' },
                  { id: 'bounty' as const, icon: Award, label: 'Bounty' },
                  { id: 'dex' as const, icon: TrendingUp, label: 'DEX' },
                ] : []),
                ...(isNodeAdmin ? [
                  { id: 'settings' as const, icon: Settings, label: 'Node Settings' },
                  { id: 'mining' as const, icon: Zap, label: 'Mining Mode' },
                  { id: 'compute' as const, icon: Cpu, label: 'Compute' },
                ] : []),
              ]).map(tab => (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id);
                    if (tab.id === 'bank') fetchBankData();
                    if (tab.id === 'settings') fetchSettingsData();
                    if (tab.id === 'bounty') fetchBountyData();
                    if (tab.id === 'dex') fetchDexData();
                    if (tab.id === 'mining') fetchMiningModeStatus();
                    if (tab.id === 'kparam') fetchKParamData();
                    if (tab.id === 'compute') fetchComputeData();
                  }}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-t-lg text-xs font-medium transition-all ${
                    activeTab === tab.id
                      ? 'bg-slate-800/60 text-emerald-300 border border-b-0 border-emerald-500/30'
                      : 'text-amber-200/50 hover:text-amber-200/80 hover:bg-slate-800/30'
                  }`}
                >
                  <tab.icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="p-5 space-y-4">
              {activeTab === 'overview' && (<>
              {/* Connection Info Bar */}
              <div className="rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-3">
                <div className="flex items-center gap-2 mb-2">
                  <MonitorSmartphone className="w-4 h-4 text-indigo-400" />
                  <span className="text-xs font-semibold text-indigo-200">Frontend Connection</span>
                </div>
                <div className="grid grid-cols-3 gap-3 text-xs">
                  {/* Active API Server */}
                  <div className="flex flex-col gap-1">
                    <span className="text-amber-200/50 text-[10px] uppercase tracking-wider">API Server</span>
                    <div className="flex items-center gap-1.5">
                      <Globe className="w-3 h-3 text-indigo-400" />
                      <span className={`font-medium ${connInfo.isPrimary ? 'text-emerald-300' : 'text-amber-300'}`}>
                        {connInfo.serverName}
                      </span>
                    </div>
                    <span className="text-amber-200/40 text-[10px] truncate" title={connInfo.activeServer}>
                      {connInfo.activeServer.replace('https://', '').replace('http://', '')}
                    </span>
                  </div>

                  {/* SSE Stream */}
                  <div className="flex flex-col gap-1">
                    <span className="text-amber-200/50 text-[10px] uppercase tracking-wider">SSE Stream</span>
                    <div className="flex items-center gap-1.5">
                      {sseStatus === 'connected' ? (
                        <>
                          <Wifi className="w-3 h-3 text-emerald-400" />
                          <span className="font-medium text-emerald-300">Connected</span>
                        </>
                      ) : sseStatus === 'reconnecting' ? (
                        <>
                          <RefreshCw className="w-3 h-3 text-amber-400 animate-spin" />
                          <span className="font-medium text-amber-300">Reconnecting</span>
                        </>
                      ) : (
                        <>
                          <WifiOff className="w-3 h-3 text-red-400" />
                          <span className="font-medium text-red-300">Disconnected</span>
                        </>
                      )}
                    </div>
                    <span className="text-amber-200/40 text-[10px]">Real-time events</span>
                  </div>

                  {/* Nginx Route */}
                  <div className="flex flex-col gap-1">
                    <span className="text-amber-200/50 text-[10px] uppercase tracking-wider">Nginx Route</span>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-amber-200/60">quillon.xyz</span>
                      <ArrowRight className="w-2.5 h-2.5 text-amber-200/40" />
                      <span className={`text-[10px] font-medium ${connInfo.isPrimary ? 'text-emerald-300' : 'text-amber-300'}`}>
                        {connInfo.isPrimary ? 'Beta:8080' : 'Gamma:8080'}
                      </span>
                    </div>
                    <span className="text-amber-200/40 text-[10px]">
                      {connInfo.isPrimary ? 'Weight 10:1' : 'Failover active'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                  <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
                  <span className="text-sm text-red-300">{error}</span>
                </div>
              )}

              {/* Server Status Cards */}
              {deployStatus ? (
                <>
                  <div className="grid grid-cols-5 gap-3">
                    <ServerCard
                      node={deployStatus.epsilon}
                      isActive={true}
                      role="supernode"
                      syncMetrics={syncMetricsMap.epsilon}
                      miningCap={miningCapacity?.epsilon}
                    />
                    <ServerCard
                      node={deployStatus.beta}
                      isActive={connInfo.isPrimary}
                      role="primary"
                      syncMetrics={syncMetricsMap.beta}
                      miningCap={miningCapacity?.beta}
                    />
                    <ServerCard
                      node={deployStatus.gamma}
                      isActive={!connInfo.isPrimary}
                      role="backup"
                      syncMetrics={syncMetricsMap.gamma}
                      miningCap={miningCapacity?.gamma}
                    />
                    <ServerCard
                      node={deployStatus.delta}
                      isActive={false}
                      role="bootstrap"
                      syncMetrics={syncMetricsMap.delta}
                      miningCap={miningCapacity?.delta}
                    />
                    <ServerCard
                      node={deployStatus.alpha}
                      isActive={false}
                      role="canary"
                      syncMetrics={syncMetricsMap.alpha}
                      miningCap={miningCapacity?.alpha}
                    />
                  </div>

                  {/* v9.0.6: Caddy Reverse Proxy Metrics */}
                  {isMasterWallet && caddyStats?.epsilon && (
                    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Globe className="w-4 h-4 text-cyan-400" />
                          <span className="text-xs font-semibold text-cyan-200"
                            title="Caddy is the reverse proxy (load balancer) that sits between users and the blockchain node. It handles TLS encryption, routes requests to the right backend service, and protects against overload. Think of it as the front door bouncer for the server.">
                            Caddy Reverse Proxy
                          </span>
                          <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                            title="Epsilon is the 10Gbit supernode (89.149.241.126) that serves quillon.xyz. It has the fastest connection and handles all public traffic.">
                            Epsilon
                          </span>
                        </div>
                        {caddyStats.epsilon.online && (
                          <div className="flex items-center gap-1"
                            title="The green pulse means Caddy's metrics endpoint (localhost:2019/metrics) is responding. If this goes dark, Caddy may have crashed or been misconfigured.">
                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                            <span className="text-[9px] text-emerald-300/70">Live</span>
                          </div>
                        )}
                      </div>

                      {caddyStats.epsilon.online ? (() => {
                        const s = caddyStats.epsilon!;
                        const totalByStatus = s.requests_by_status.ok_2xx + s.requests_by_status.client_err_4xx + s.requests_by_status.server_err_5xx + s.requests_by_status.redirect_3xx + s.requests_by_status.websocket_101;
                        const errRate = totalByStatus > 0 ? ((s.requests_by_status.server_err_5xx / totalByStatus) * 100) : 0;
                        return (
                          <>
                            {/* Top row: key gauges */}
                            <div className="grid grid-cols-5 gap-2 mb-3">
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Requests per second (req/s)\n\nHow many HTTP requests Caddy handles every second, like a cashier counting customers. This includes miners submitting work, wallets checking balances, and SSE streams.\n\nGreen = healthy load\nAmber (>500) = heavy traffic, watch for bottlenecks\n\nCurrent: " + (s.requests_per_second > 0 ? s.requests_per_second.toFixed(1) : '0') + " req/s"}>
                                <div className={`text-base font-bold ${
                                  s.requests_per_second > 500 ? 'text-amber-300' : 'text-cyan-300'
                                }`}>
                                  {s.requests_per_second > 0 ? s.requests_per_second.toFixed(0) : '0'}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">req/s</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Average Response Time (ms)\n\nHow long it takes Caddy to respond to a request, measured in milliseconds (1000ms = 1 second). This is the mean across ALL requests.\n\nGreen (<100ms) = snappy, users won't notice any delay\nAmber (100-500ms) = acceptable, some slowness\nRed (>500ms) = slow, users will feel lag\n\nLike measuring how fast a waiter brings your food. Lower is better.\n\nCurrent: " + s.avg_response_ms.toFixed(1) + "ms"}>
                                <div className={`text-base font-bold ${
                                  s.avg_response_ms > 500 ? 'text-red-400' :
                                  s.avg_response_ms > 100 ? 'text-amber-300' : 'text-emerald-300'
                                }`}>
                                  {s.avg_response_ms < 1 ? '<1' : s.avg_response_ms.toFixed(0)}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">avg ms</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"99th Percentile Response Time (p99)\n\nThe slowest 1% of requests take at least this long. If p99 = 500ms, then 99 out of 100 requests finish faster than 500ms. Only 1 in 100 is slower.\n\nThis catches worst-case performance that the average hides. A good average with a bad p99 means most users are happy but some are getting terrible performance.\n\nGreen (<500ms) = even the slowest requests are fast\nAmber (500ms-2s) = some users hitting delays\nRed (>2s) = tail latency problem, investigate\n\nCurrent: " + s.p99_response_ms.toFixed(1) + "ms"}>
                                <div className={`text-base font-bold ${
                                  s.p99_response_ms > 2000 ? 'text-red-400' :
                                  s.p99_response_ms > 500 ? 'text-amber-300' : 'text-emerald-300'
                                }`}>
                                  {s.p99_response_ms < 1 ? '<1' : s.p99_response_ms >= 1000 ? `${(s.p99_response_ms / 1000).toFixed(1)}s` : `${s.p99_response_ms.toFixed(0)}`}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">p99 ms</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Server Error Rate (5xx)\n\nPercentage of requests that failed with a server error (HTTP 500-599). These are OUR fault, not the user's.\n\n5xx errors mean the backend couldn't handle the request: overloaded, crashed, timed out, or hit a bug.\n\nGreen (<1%) = healthy, rare errors\nAmber (1-5%) = concerning, some users affected\nRed (>5%) = critical, many requests failing\n\nCommon causes: mining semaphore full (503), node syncing, backend restart.\n\nCurrent: " + errRate.toFixed(2) + "% (" + s.requests_by_status.server_err_5xx.toLocaleString() + " of " + totalByStatus.toLocaleString() + " total)"}>
                                <div className={`text-base font-bold ${
                                  errRate > 5 ? 'text-red-400' :
                                  errRate > 1 ? 'text-amber-300' : 'text-emerald-300'
                                }`}>
                                  {errRate > 0 ? errRate.toFixed(1) : '0'}%
                                </div>
                                <div className="text-[9px] text-cyan-300/50">5xx err</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Goroutines (concurrent tasks)\n\nGoroutines are lightweight threads inside Caddy (written in Go). Each active connection, request, or background task uses one or more goroutines.\n\nThink of them like workers in a factory. More workers = more concurrent activity.\n\n1-5K = normal idle\n10-50K = moderate load (hundreds of miners)\n50K+ = heavy load, check if connections are leaking\n\nUnlike OS threads, Go can handle millions of goroutines. High numbers aren't necessarily bad, but a steadily rising count without more traffic suggests a connection leak.\n\nCurrent: " + s.goroutines.toLocaleString()}>
                                <div className="text-base font-bold text-blue-300">
                                  {s.goroutines > 1000 ? `${(s.goroutines / 1000).toFixed(1)}K` : s.goroutines}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">goroutines</div>
                              </div>
                            </div>

                            {/* Status code breakdown bar */}
                            <div className="mb-3">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-[9px] text-cyan-300/50 uppercase tracking-wider cursor-help"
                                  title={"HTTP Response Code Breakdown\n\nEvery HTTP response has a status code. This bar shows the distribution:\n\n2xx (green) = Success. The request worked perfectly.\n3xx (blue) = Redirect. The client was sent to a different URL.\n4xx (amber) = Client error. Bad request, not found, unauthorized. The user did something wrong.\n5xx (red) = Server error. The server failed. Our problem to fix.\n101 WS (purple) = WebSocket upgrade. A persistent two-way connection was established (used for SSE mining streams).\n\nA healthy server has mostly green (2xx). Some 5xx is normal under heavy mining load (503 = semaphore full, miner retries automatically)."}>
                                  Response Codes
                                </span>
                                <span className="text-[9px] text-cyan-300/40 cursor-help"
                                  title={"Total requests handled by Caddy since it was last started or reloaded. This counter resets on Caddy restart.\n\nTotal: " + s.total_requests.toLocaleString()}>
                                  {s.total_requests > 1000000 ? `${(s.total_requests / 1000000).toFixed(1)}M` :
                                   s.total_requests > 1000 ? `${(s.total_requests / 1000).toFixed(1)}K` :
                                   s.total_requests} total
                                </span>
                              </div>
                              {totalByStatus > 0 && (
                                <div className="h-2 rounded-full overflow-hidden flex bg-slate-800/60">
                                  {s.requests_by_status.ok_2xx > 0 && (
                                    <div className="bg-emerald-500 h-full cursor-help" style={{ width: `${(s.requests_by_status.ok_2xx / totalByStatus) * 100}%` }}
                                      title={`2xx Success: ${s.requests_by_status.ok_2xx.toLocaleString()} requests (${((s.requests_by_status.ok_2xx / totalByStatus) * 100).toFixed(1)}%)\n\nThese requests completed successfully. The server understood the request and returned the expected data. This is the ideal outcome for every request.`} />
                                  )}
                                  {s.requests_by_status.redirect_3xx > 0 && (
                                    <div className="bg-blue-500 h-full cursor-help" style={{ width: `${(s.requests_by_status.redirect_3xx / totalByStatus) * 100}%` }}
                                      title={`3xx Redirect: ${s.requests_by_status.redirect_3xx.toLocaleString()} requests (${((s.requests_by_status.redirect_3xx / totalByStatus) * 100).toFixed(1)}%)\n\nThe client was told to go to a different URL. Common for HTTP->HTTPS upgrades (301) or temporary redirects (302). Normal and expected.`} />
                                  )}
                                  {s.requests_by_status.client_err_4xx > 0 && (
                                    <div className="bg-amber-500 h-full cursor-help" style={{ width: `${(s.requests_by_status.client_err_4xx / totalByStatus) * 100}%` }}
                                      title={`4xx Client Error: ${s.requests_by_status.client_err_4xx.toLocaleString()} requests (${((s.requests_by_status.client_err_4xx / totalByStatus) * 100).toFixed(1)}%)\n\nThe client sent a bad request. Examples:\n- 400 Bad Request (malformed data)\n- 401 Unauthorized (no login)\n- 403 Forbidden (wrong wallet)\n- 404 Not Found (wrong URL)\n\nSmall numbers are normal (bots, typos). A spike could mean a broken client update.`} />
                                  )}
                                  {s.requests_by_status.server_err_5xx > 0 && (
                                    <div className="bg-red-500 h-full cursor-help" style={{ width: `${(s.requests_by_status.server_err_5xx / totalByStatus) * 100}%` }}
                                      title={`5xx Server Error: ${s.requests_by_status.server_err_5xx.toLocaleString()} requests (${((s.requests_by_status.server_err_5xx / totalByStatus) * 100).toFixed(1)}%)\n\nThe server failed to handle the request. Common causes:\n- 502 Bad Gateway (backend crashed or unreachable)\n- 503 Service Unavailable (mining semaphore full, too many concurrent submissions)\n- 504 Gateway Timeout (backend took too long)\n\nSome 503s during heavy mining are expected. The miner retries automatically. A sudden spike in 502s means the backend process may have crashed.`} />
                                  )}
                                  {s.requests_by_status.websocket_101 > 0 && (
                                    <div className="bg-purple-500 h-full cursor-help" style={{ width: `${(s.requests_by_status.websocket_101 / totalByStatus) * 100}%` }}
                                      title={`101 WebSocket Upgrade: ${s.requests_by_status.websocket_101.toLocaleString()} connections\n\nWebSocket upgrades create a persistent two-way connection. Used for:\n- SSE (Server-Sent Events) mining reward streams\n- Real-time balance updates\n- Live block notifications\n\nEach connected wallet/miner holds one WebSocket. Low numbers are normal since these are long-lived connections, not individual requests.`} />
                                  )}
                                </div>
                              )}
                              <div className="flex gap-3 mt-1">
                                {[
                                  { label: '2xx', count: s.requests_by_status.ok_2xx, color: 'text-emerald-400', tip: 'Successful responses' },
                                  { label: '3xx', count: s.requests_by_status.redirect_3xx, color: 'text-blue-400', tip: 'Redirects (HTTP->HTTPS etc.)' },
                                  { label: '4xx', count: s.requests_by_status.client_err_4xx, color: 'text-amber-400', tip: 'Client errors (bad request, not found)' },
                                  { label: '5xx', count: s.requests_by_status.server_err_5xx, color: 'text-red-400', tip: 'Server errors (overload, crash, timeout)' },
                                  { label: 'WS', count: s.requests_by_status.websocket_101, color: 'text-purple-400', tip: 'WebSocket upgrades (live connections)' },
                                ].filter(x => x.count > 0).map(x => (
                                  <span key={x.label} className={`text-[9px] ${x.color} cursor-help`} title={`${x.tip}: ${x.count.toLocaleString()}`}>
                                    {x.label}: {x.count > 1000000 ? `${(x.count / 1000000).toFixed(1)}M` : x.count > 1000 ? `${(x.count / 1000).toFixed(1)}K` : x.count}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Bottom row: memory + upstreams */}
                            <div className="flex items-center justify-between text-[9px] text-cyan-300/40">
                              <span className="cursor-help"
                                title={"Caddy Heap Memory: " + s.memory_mb.toFixed(1) + " MB\n\nHow much RAM Caddy's Go runtime is using for its heap (dynamic allocations). This includes connection buffers, TLS session caches, and request/response data.\n\nNormal: 100-500 MB\nHigh: 500-2000 MB (lots of concurrent connections)\nCritical: >2000 MB (possible memory leak or connection pileup)\n\nCaddy's Go garbage collector reclaims unused memory automatically, but under heavy load memory stays high because connections are actively using it."}>
                                Heap: {s.memory_mb.toFixed(0)} MB
                              </span>
                              <div className="flex gap-2">
                                {s.upstreams.map(u => (
                                  <span key={u.address} className="flex items-center gap-1 cursor-help"
                                    title={`Upstream: ${u.address}\nStatus: ${u.healthy ? 'Healthy' : 'DOWN'}\n\nUpstreams are the backend services Caddy forwards requests to. Each one runs a different part of the system:\n- localhost:8080 = blockchain node (API, mining, P2P)\n- localhost:3080 = bounty server\n- localhost:9002 = additional service\n\nGreen dot = Caddy can reach this backend and it responds to health checks.\nRed dot = Backend is unreachable or failing health checks. Caddy will stop routing traffic to it until it recovers.`}>
                                    <div className={`w-1.5 h-1.5 rounded-full ${u.healthy ? 'bg-emerald-400' : 'bg-red-400'}`} />
                                    <span className={u.healthy ? 'text-emerald-300/70' : 'text-red-300/70'}>{u.address}</span>
                                  </span>
                                ))}
                              </div>
                            </div>
                          </>
                        );
                      })() : (
                        <div className="text-[10px] text-red-400/60"
                          title="Caddy's Prometheus metrics endpoint (http://localhost:2019/metrics) is not responding. This could mean Caddy is not running, crashed, or the admin API port (2019) is blocked. Check: systemctl status caddy">
                          Caddy metrics offline
                        </div>
                      )}
                    </div>
                  )}

                  {/* v9.2.0: q-flux Reverse Proxy Metrics */}
                  {isMasterWallet && fluxStats?.online && (
                    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Zap className="w-4 h-4 text-cyan-400" />
                          <span className="text-xs font-semibold text-cyan-200"
                            title="q-flux is a worker-per-core TLS reverse proxy written in Rust. It handles TLS termination, HTTP/2 multiplexing, WebSocket upgrades, and upstream load balancing with minimal latency. Each CPU core runs its own event loop for zero-contention request handling.">
                            q-flux Reverse Proxy
                          </span>
                          {fluxStats.version && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                              title="q-flux binary version">
                              v{fluxStats.version}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1"
                          title="The green pulse means q-flux admin (127.0.0.1:9090/status) is responding. If this goes dark, q-flux may have crashed.">
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="text-[9px] text-emerald-300/70">Live</span>
                        </div>
                      </div>

                      {(() => {
                        const f = fluxStats;
                        const totalByStatus = f.requests_2xx + f.requests_4xx + f.requests_5xx;
                        const h2Active = f.h2_streams_opened - f.h2_streams_closed;
                        return (
                          <>
                            {/* Row 1: 6 stat gauges */}
                            <div className="grid grid-cols-6 gap-2 mb-3">
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Requests per second (req/s)\n\nHow many HTTP requests q-flux handles every second. Computed from delta between poll intervals.\n\nCyan = normal\nAmber (>500) = heavy traffic\n\nCurrent: " + (f.requests_per_second > 0 ? f.requests_per_second.toFixed(1) : '0') + " req/s"}>
                                <div className={`text-base font-bold ${f.requests_per_second > 500 ? 'text-amber-300' : 'text-cyan-300'}`}>
                                  {f.requests_per_second > 0 ? f.requests_per_second.toFixed(0) : '0'}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">req/s</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Active Connections\n\nCurrently open TCP connections to q-flux. Each miner, wallet, or SSE stream holds one connection.\n\nCurrent: " + f.active_connections.toLocaleString()}>
                                <div className="text-base font-bold text-cyan-300">
                                  {f.active_connections > 1000 ? `${(f.active_connections / 1000).toFixed(1)}K` : f.active_connections}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">Conns</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Worker Threads\n\nq-flux runs one worker per CPU core. Each worker has its own event loop for zero-contention request handling. More workers = more parallel capacity.\n\nCurrent: " + f.worker_count}>
                                <div className="text-base font-bold text-blue-300">{f.worker_count}</div>
                                <div className="text-[9px] text-cyan-300/50">Workers</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Active WebSocket Connections\n\nLive WebSocket/SSE streams. Each connected wallet or miner may hold one persistent WebSocket for real-time updates.\n\nTotal upgrades since start: " + f.websocket_upgrades.toLocaleString() + "\nCurrent active: " + f.active_websockets.toLocaleString()}>
                                <div className="text-base font-bold text-purple-300">{f.active_websockets}</div>
                                <div className="text-[9px] text-cyan-300/50">WS</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Active HTTP/2 Streams\n\nConcurrent HTTP/2 streams (opened minus closed). HTTP/2 multiplexes many requests over a single TCP connection, reducing overhead.\n\nOpened: " + f.h2_streams_opened.toLocaleString() + "\nClosed: " + f.h2_streams_closed.toLocaleString() + "\nActive: " + h2Active.toLocaleString()}>
                                <div className="text-base font-bold text-emerald-300">
                                  {h2Active > 1000 ? `${(h2Active / 1000).toFixed(1)}K` : h2Active}
                                </div>
                                <div className="text-[9px] text-cyan-300/50">H2 Streams</div>
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 text-center cursor-help"
                                title={"Server Error Rate (5xx%)\n\nPercentage of responses that were server errors (500-599). These indicate backend failures.\n\nGreen (<1%) = healthy\nAmber (1-5%) = concerning\nRed (>5%) = critical\n\nCurrent: " + f.error_rate_pct.toFixed(2) + "% (" + f.requests_5xx.toLocaleString() + " of " + totalByStatus.toLocaleString() + " total)"}>
                                <div className={`text-base font-bold ${
                                  f.error_rate_pct > 5 ? 'text-red-400' :
                                  f.error_rate_pct > 1 ? 'text-amber-300' : 'text-emerald-300'
                                }`}>
                                  {f.error_rate_pct > 0 ? f.error_rate_pct.toFixed(1) : '0'}%
                                </div>
                                <div className="text-[9px] text-cyan-300/50">5xx err</div>
                              </div>
                            </div>

                            {/* Row 2: Response code breakdown bar */}
                            <div className="mb-3">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-[9px] text-cyan-300/50 uppercase tracking-wider cursor-help"
                                  title="HTTP response code breakdown from q-flux. Green = success (2xx), Amber = client error (4xx), Red = server error (5xx).">
                                  Response Codes
                                </span>
                                <span className="text-[9px] text-cyan-300/40 cursor-help"
                                  title={"Total requests handled by q-flux since start.\n\nTotal: " + f.total_requests.toLocaleString()}>
                                  {f.total_requests > 1000000 ? `${(f.total_requests / 1000000).toFixed(1)}M` :
                                   f.total_requests > 1000 ? `${(f.total_requests / 1000).toFixed(1)}K` :
                                   f.total_requests} total
                                </span>
                              </div>
                              {totalByStatus > 0 && (
                                <div className="h-2 rounded-full overflow-hidden flex bg-slate-800/60">
                                  {f.requests_2xx > 0 && (
                                    <div className="bg-emerald-500 h-full cursor-help" style={{ width: `${(f.requests_2xx / totalByStatus) * 100}%` }}
                                      title={`2xx Success: ${f.requests_2xx.toLocaleString()} (${((f.requests_2xx / totalByStatus) * 100).toFixed(1)}%)`} />
                                  )}
                                  {f.requests_4xx > 0 && (
                                    <div className="bg-amber-500 h-full cursor-help" style={{ width: `${(f.requests_4xx / totalByStatus) * 100}%` }}
                                      title={`4xx Client Error: ${f.requests_4xx.toLocaleString()} (${((f.requests_4xx / totalByStatus) * 100).toFixed(1)}%)`} />
                                  )}
                                  {f.requests_5xx > 0 && (
                                    <div className="bg-red-500 h-full cursor-help" style={{ width: `${(f.requests_5xx / totalByStatus) * 100}%` }}
                                      title={`5xx Server Error: ${f.requests_5xx.toLocaleString()} (${((f.requests_5xx / totalByStatus) * 100).toFixed(1)}%)`} />
                                  )}
                                </div>
                              )}
                              <div className="flex gap-3 mt-1">
                                {[
                                  { label: '2xx', count: f.requests_2xx, color: 'text-emerald-400' },
                                  { label: '4xx', count: f.requests_4xx, color: 'text-amber-400' },
                                  { label: '5xx', count: f.requests_5xx, color: 'text-red-400' },
                                ].filter(x => x.count > 0).map(x => (
                                  <span key={x.label} className={`text-[9px] ${x.color}`}>
                                    {x.label}: {x.count > 1000000 ? `${(x.count / 1000000).toFixed(1)}M` : x.count > 1000 ? `${(x.count / 1000).toFixed(1)}K` : x.count}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Row 3: TLS / Upstream / Bandwidth info cards */}
                            <div className="grid grid-cols-3 gap-2 mb-3">
                              <div className="bg-slate-800/40 rounded-lg p-2 cursor-help"
                                title={"TLS Statistics\n\nHandshakes OK: " + f.tls_handshakes.toLocaleString() + "\nHandshake Failures: " + f.tls_handshake_failures.toLocaleString() + "\nCert Reloads: " + f.tls_reload_count + "\nFailure Rate: " + (f.tls_handshakes > 0 ? ((f.tls_handshake_failures / (f.tls_handshakes + f.tls_handshake_failures)) * 100).toFixed(2) : '0') + "%\n\nTLS handshake failures can indicate:\n- Expired certificates\n- Incompatible cipher suites\n- Client-side issues (old browsers, bots)"}>
                                <div className="text-[9px] text-cyan-300/50 mb-1">TLS</div>
                                <div className="text-[10px] text-cyan-200">
                                  <span className="text-emerald-300">{f.tls_handshakes > 1000 ? `${(f.tls_handshakes / 1000).toFixed(1)}K` : f.tls_handshakes}</span>
                                  {f.tls_handshake_failures > 0 && <span className="text-red-300"> / {f.tls_handshake_failures} fail</span>}
                                </div>
                                {f.tls_reload_count > 0 && <div className="text-[9px] text-cyan-300/40">{f.tls_reload_count} reload{f.tls_reload_count !== 1 ? 's' : ''}</div>}
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 cursor-help"
                                title={"Upstream Backend\n\nActive connections to backend: " + f.upstream_active + "\nConnect failures: " + f.upstream_connect_failures.toLocaleString() + "\nTimeouts: " + f.upstream_timeouts.toLocaleString() + "\n\nConnect failures mean q-flux couldn't reach the backend (port 8080). Timeouts mean the backend took too long to respond."}>
                                <div className="text-[9px] text-cyan-300/50 mb-1">Upstream</div>
                                <div className="text-[10px] text-cyan-200">
                                  <span className="text-emerald-300">{f.upstream_active}</span> active
                                </div>
                                {(f.upstream_connect_failures > 0 || f.upstream_timeouts > 0) && (
                                  <div className="text-[9px] text-amber-300/70">
                                    {f.upstream_connect_failures > 0 && <span>{f.upstream_connect_failures} fail </span>}
                                    {f.upstream_timeouts > 0 && <span>{f.upstream_timeouts} timeout</span>}
                                  </div>
                                )}
                              </div>
                              <div className="bg-slate-800/40 rounded-lg p-2 cursor-help"
                                title={"Bandwidth\n\nBytes received (from clients): " + f.bytes_received.toLocaleString() + "\nBytes sent (to clients): " + f.bytes_sent.toLocaleString() + "\n\nTotal data transferred through q-flux since start."}>
                                <div className="text-[9px] text-cyan-300/50 mb-1">Bandwidth</div>
                                <div className="text-[10px] text-cyan-200">
                                  <span className="text-blue-300">↓{f.bytes_received > 1073741824 ? `${(f.bytes_received / 1073741824).toFixed(1)}GB` : f.bytes_received > 1048576 ? `${(f.bytes_received / 1048576).toFixed(0)}MB` : `${(f.bytes_received / 1024).toFixed(0)}KB`}</span>
                                  {' '}
                                  <span className="text-emerald-300">↑{f.bytes_sent > 1073741824 ? `${(f.bytes_sent / 1073741824).toFixed(1)}GB` : f.bytes_sent > 1048576 ? `${(f.bytes_sent / 1048576).toFixed(0)}MB` : `${(f.bytes_sent / 1024).toFixed(0)}KB`}</span>
                                </div>
                              </div>
                            </div>

                            {/* Row 4: Super-Cluster Topology */}
                            {f.cluster?.enabled && (
                              <div className="mb-3 rounded-lg bg-slate-800/40 p-2.5 border border-cyan-500/20">
                                <div className="flex items-center gap-1.5 mb-2">
                                  <svg className="w-3.5 h-3.5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="12" cy="12" r="3" /><circle cx="4" cy="6" r="2" /><circle cx="20" cy="6" r="2" /><circle cx="4" cy="18" r="2" /><circle cx="20" cy="18" r="2" />
                                    <line x1="6" y1="7" x2="10" y2="10" /><line x1="18" y1="7" x2="14" y2="10" /><line x1="6" y1="17" x2="10" y2="14" /><line x1="18" y1="17" x2="14" y2="14" />
                                  </svg>
                                  <span className="text-[10px] font-semibold text-cyan-300 cursor-help"
                                    title="Super-Cluster: Cross-node failover mesh. q-flux routes requests to local backends first. If ALL local backends are down, traffic automatically fails over to cluster peers (remote servers). This provides zero-downtime resilience across the entire server fleet.">
                                    Super-Cluster Topology
                                  </span>
                                  <span className="text-[9px] px-1 py-0.5 rounded bg-cyan-500/15 text-cyan-400/80 border border-cyan-500/20 cursor-help"
                                    title={`${f.cluster.local_backends.length} local backend(s) + ${f.cluster.cluster_peers.length} cluster peer(s) in the mesh`}>
                                    {f.cluster.local_backends.length + f.cluster.cluster_peers.length} nodes
                                  </span>
                                </div>

                                {/* Visual topology: Local → this node → Cluster peers */}
                                <div className="flex items-center gap-1.5 flex-wrap">
                                  {/* Local backends */}
                                  {f.cluster.local_backends.map((b, i) => {
                                    const label = b.addr.replace('127.0.0.1:', 'localhost:');
                                    return (
                                      <div key={`local-${i}`} className={`flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-mono border cursor-help ${
                                        b.healthy
                                          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                                          : 'bg-red-500/10 border-red-500/30 text-red-300'
                                      }`}
                                        title={`Local Backend: ${b.addr}\nStatus: ${b.healthy ? 'HEALTHY' : 'UNHEALTHY'}\nConsecutive Failures: ${b.failures}\nLast Health Check: ${b.last_check_ms_ago < 1000 ? '<1s' : Math.floor(b.last_check_ms_ago / 1000) + 's'} ago\n\nLocal backends are the q-api-server instances on this machine. q-flux always prefers local backends for lowest latency.`}>
                                        <div className={`w-1.5 h-1.5 rounded-full ${b.healthy ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
                                        <span>{label}</span>
                                        <span className="text-[8px] opacity-50">LOCAL</span>
                                      </div>
                                    );
                                  })}

                                  {/* Arrow connector */}
                                  {f.cluster.cluster_peers.length > 0 && (
                                    <div className="flex items-center gap-0.5 text-cyan-500/40 cursor-help"
                                      title="Failover direction: if all local backends are unhealthy, q-flux routes to cluster peers. Traffic always prefers local first for lowest latency.">
                                      <div className="w-4 h-px bg-cyan-500/30" />
                                      <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M5 12h14m-7-7l7 7-7 7" />
                                      </svg>
                                      <span className="text-[8px]">failover</span>
                                      <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M5 12h14m-7-7l7 7-7 7" />
                                      </svg>
                                      <div className="w-4 h-px bg-cyan-500/30" />
                                    </div>
                                  )}

                                  {/* Cluster peers */}
                                  {f.cluster.cluster_peers.map((p, i) => {
                                    // Try to resolve addr to server name
                                    const serverName = p.addr.includes('89.149.241.126') ? 'Epsilon'
                                      : p.addr.includes('185.182.185.227') ? 'Beta'
                                      : p.addr.includes('109.205.176.60') ? 'Gamma'
                                      : p.addr.includes('5.79.79.158') ? 'Delta'
                                      : p.addr.split(':')[0];
                                    return (
                                      <div key={`cluster-${i}`} className={`flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-mono border cursor-help ${
                                        p.healthy
                                          ? 'bg-blue-500/10 border-blue-500/30 text-blue-300'
                                          : 'bg-red-500/10 border-red-500/30 text-red-300'
                                      }`}
                                        title={`Cluster Peer: ${p.addr}\nServer: ${serverName}\nStatus: ${p.healthy ? 'HEALTHY' : 'UNHEALTHY'}\nConsecutive Failures: ${p.failures}\nLast Health Check: ${p.last_check_ms_ago < 1000 ? '<1s' : Math.floor(p.last_check_ms_ago / 1000) + 's'} ago\n\nCluster peers are remote q-api-server instances on other physical servers. q-flux only routes to them when ALL local backends are unhealthy (failover mode).`}>
                                        <div className={`w-1.5 h-1.5 rounded-full ${p.healthy ? 'bg-blue-400 animate-pulse' : 'bg-red-400'}`} />
                                        <span>{serverName}</span>
                                        <span className="text-[8px] opacity-50">PEER</span>
                                      </div>
                                    );
                                  })}
                                </div>

                                {/* Status summary */}
                                {(() => {
                                  const allLocal = f.cluster!.local_backends;
                                  const allPeers = f.cluster!.cluster_peers;
                                  const localHealthy = allLocal.filter(b => b.healthy).length;
                                  const peersHealthy = allPeers.filter(p => p.healthy).length;
                                  const allLocalDown = localHealthy === 0 && allLocal.length > 0;
                                  return (
                                    <div className={`mt-1.5 text-[9px] flex items-center gap-1.5 ${allLocalDown ? 'text-amber-300' : 'text-cyan-300/50'}`}>
                                      {allLocalDown ? (
                                        <>
                                          <svg className="w-3 h-3 text-amber-400 animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4m0 4h.01" />
                                          </svg>
                                          <span>FAILOVER ACTIVE — routing to cluster peers ({peersHealthy}/{allPeers.length} healthy)</span>
                                        </>
                                      ) : (
                                        <span>Local: {localHealthy}/{allLocal.length} healthy | Peers: {peersHealthy}/{allPeers.length} healthy</span>
                                      )}
                                    </div>
                                  );
                                })()}
                              </div>
                            )}

                            {/* Row 5: Bottom metadata row */}
                            <div className="flex items-center justify-between text-[9px] text-cyan-300/40">
                              <span className="cursor-help"
                                title={"q-flux uptime since last start.\n\nUptime: " + f.uptime_secs + " seconds"}>
                                Up: {f.uptime_secs >= 86400 ? `${Math.floor(f.uptime_secs / 86400)}d ${Math.floor((f.uptime_secs % 86400) / 3600)}h` : f.uptime_secs >= 3600 ? `${Math.floor(f.uptime_secs / 3600)}h ${Math.floor((f.uptime_secs % 3600) / 60)}m` : `${Math.floor(f.uptime_secs / 60)}m`}
                              </span>
                              <span className="cursor-help"
                                title={"Total connections accepted since start: " + f.total_connections.toLocaleString()}>
                                Total conns: {f.total_connections > 1000000 ? `${(f.total_connections / 1000000).toFixed(1)}M` : f.total_connections > 1000 ? `${(f.total_connections / 1000).toFixed(1)}K` : f.total_connections}
                              </span>
                              {f.rate_limited > 0 && (
                                <span className="cursor-help text-amber-300/70"
                                  title={"Requests rejected by rate limiter: " + f.rate_limited.toLocaleString() + "\n\nThese requests were blocked because a client exceeded the allowed request rate. This protects the backend from DDoS or runaway clients."}>
                                  Rate limited: {f.rate_limited > 1000 ? `${(f.rate_limited / 1000).toFixed(1)}K` : f.rate_limited}
                                </span>
                              )}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  )}

                  {/* Decentralization Index Gauge */}
                  {decentral && (
                    <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Globe className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm font-bold text-cyan-200 tracking-wide">DECENTRALIZATION INDEX</span>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-lg font-black ${
                          decentral.grade.startsWith('A') ? 'bg-emerald-500/20 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                          : decentral.grade === 'B' ? 'bg-blue-500/20 text-blue-300 shadow-[0_0_12px_rgba(59,130,246,0.3)]'
                          : decentral.grade === 'C' ? 'bg-amber-500/20 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.3)]'
                          : 'bg-red-500/20 text-red-300 shadow-[0_0_12px_rgba(239,68,68,0.3)]'
                        }`}>
                          {decentral.grade}
                        </div>
                      </div>

                      {/* Main progress bar */}
                      <div className="relative h-4 rounded-full bg-slate-700/50 overflow-hidden mb-4">
                        <div
                          className="h-full rounded-full transition-all duration-1000 relative overflow-hidden"
                          style={{
                            width: `${decentral.decentralization_index}%`,
                            background: decentral.decentralization_index >= 75
                              ? 'linear-gradient(90deg, #10b981, #06b6d4)'
                              : decentral.decentralization_index >= 40
                              ? 'linear-gradient(90deg, #f59e0b, #10b981)'
                              : 'linear-gradient(90deg, #ef4444, #f59e0b)',
                          }}
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[shimmer_2s_infinite]"
                            style={{ animation: 'shimmer 2s infinite linear', backgroundSize: '200% 100%' }} />
                        </div>
                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] font-bold text-white/80">
                          {decentral.decentralization_index.toFixed(0)}/100
                        </span>
                      </div>

                      {/* Sub-metric cards (7 cards: sqrt scaling matches backend) */}
                      <div className="grid grid-cols-7 gap-2 mb-3">
                        {[
                          { label: 'Nakamoto', display: String(decentral.nakamoto_coefficient), pct: Math.min(Math.sqrt(decentral.nakamoto_coefficient / 10) * 100, 100) },
                          { label: 'Mine Gini', display: decentral.gini_coefficient.toFixed(2), pct: (1 - decentral.gini_coefficient) * 100 },
                          { label: 'Wealth Gini', display: (decentral.wealth_gini ?? 0).toFixed(2), pct: (1 - (decentral.wealth_gini ?? 0)) * 100 },
                          { label: 'Entropy', display: `${(decentral.entropy_score ?? 0).toFixed(0)}%`, pct: decentral.entropy_score ?? 0 },
                          { label: 'Miners', display: String(decentral.unique_wallets), pct: Math.min(Math.sqrt(decentral.unique_wallets / 100) * 100, 100) },
                          { label: 'Nodes', display: `${decentral.infrastructure_nodes ?? decentral.node_count}+${decentral.community_nodes ?? 0}`, pct: Math.min(Math.sqrt(decentral.node_count / 20) * 100, 100) },
                          { label: 'Peers', display: String(decentral.peer_count), pct: Math.min(Math.sqrt(decentral.peer_count / 100) * 100, 100) },
                        ].map(m => (
                          <div key={m.label} className="rounded-lg bg-slate-800/50 border border-slate-700/50 p-2 text-center">
                            <div className="text-[9px] text-slate-400 mb-1 truncate">{m.label}</div>
                            <div className="text-xs font-bold text-white mb-1">{m.display}</div>
                            <div className="h-1.5 rounded-full bg-slate-700/50 overflow-hidden mb-1">
                              <div className={`h-full rounded-full transition-all duration-700 ${
                                m.pct >= 75 ? 'bg-emerald-500' : m.pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
                              }`} style={{ width: `${m.pct}%` }} />
                            </div>
                            <div className="text-[8px] text-slate-500">{m.pct.toFixed(0)}%</div>
                          </div>
                        ))}
                      </div>

                      {/* Footer stats */}
                      <div className="flex items-center justify-center gap-4 text-[10px] text-slate-400 flex-wrap">
                        <span>Top miner: <span className="text-white/70">{decentral.top_miner_pct.toFixed(1)}%</span></span>
                        <span className="text-slate-600">·</span>
                        <span>Top 3: <span className="text-white/70">{decentral.top3_miners_pct.toFixed(1)}%</span></span>
                        <span className="text-slate-600">·</span>
                        <span>HHI: <span className="text-white/70">{decentral.hhi.toFixed(0)}</span>
                          <span className="ml-1">({decentral.hhi < 1500 ? 'unconcentrated' : decentral.hhi < 2500 ? 'moderate' : 'concentrated'})</span>
                        </span>
                        <span className="text-slate-600">·</span>
                        <span>{decentral.total_workers} workers</span>
                        <span className="text-slate-600">·</span>
                        <span>Raw: <span className="text-white/70">{(decentral.decentralization_index_raw ?? decentral.decentralization_index).toFixed(1)}</span></span>
                      </div>
                    </div>
                  )}

                  {/* CCC Convergence Readiness Panel */}
                  {convergence && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className={`rounded-xl border p-3 ${getPhaseInfo(convergence.cosmic_phase).bgGlow}`}
                      style={{
                        background: 'linear-gradient(135deg, rgba(15, 10, 40, 0.9) 0%, rgba(20, 15, 50, 0.9) 100%)',
                        borderColor: convergence.convergence_safe ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)',
                      }}
                    >
                      {/* Phase Header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <KTooltip wide text={(() => {
                            const p = getPhaseInfo(convergence.cosmic_phase).name;
                            if (p === 'Isolation') return `Isolation Phase — Each server is operating independently, like separate galaxies drifting through space before they discover each other. In cosmology, this mirrors the era before gravitational attraction pulls matter together. During this deployment phase, Alpha and Delta are receiving the new binary in parallel but haven't yet verified compatibility with the primary network. No traffic is being shifted yet — the servers are "isolated" test environments. Think of it like running an experiment in a sealed lab before releasing results to the world.`;
                            if (p === 'Convergence') return `Convergence Phase — The servers are beginning to find each other and synchronize, like galaxies drawn together by gravity. In physics, convergence describes systems approaching a stable equilibrium point. During this deployment phase, Gamma has received the new binary and is actively syncing its blockchain state with Beta (the primary). The system is verifying that the new version produces identical consensus results. This is the critical "trust but verify" stage — Gamma must prove it can handle real-world traffic before the network commits.`;
                            if (p === 'Aeon Transition') return `Aeon Transition Phase — A conformal boundary crossing, like the moment a star collapses into a new state of matter. In Roger Penrose's Conformal Cyclic Cosmology, an "aeon transition" is the boundary between one universe-epoch and the next. During this deployment phase, Beta (the primary production server) is being upgraded. Traffic has been shifted to Gamma, and Beta is crossing the boundary from old-version to new-version. This is the most delicate moment — like performing heart surgery while the patient is still alive. The network continues serving users through Gamma while Beta transforms.`;
                            if (p === 'Harmony') return `Harmony Phase — All servers have converged into a unified, synchronized state — like a solar system where all planets orbit in resonance. In physics, harmonic resonance occurs when oscillating systems naturally synchronize their frequencies. During this deployment phase, all 5 servers (Epsilon, Alpha, Beta, Gamma, Delta) are running the same version, synced to the same blockchain height, and serving traffic together. This is the ideal end-state: maximum redundancy, zero version mismatch, and the network is at peak resilience. The cosmic gardener's garden is in full bloom.`;
                            return `Cosmic Phase — The current stage of the deployment lifecycle, modeled after phases of cosmic evolution. Each phase represents a different level of network convergence and stability.`;
                          })()}>
                            <span className="text-lg cursor-help">{getPhaseInfo(convergence.cosmic_phase).icon}</span>
                          </KTooltip>
                          <div>
                            <div className="flex items-center gap-1.5">
                              <span className={`text-xs font-bold uppercase tracking-wider ${getPhaseInfo(convergence.cosmic_phase).color}`}>
                                {getPhaseInfo(convergence.cosmic_phase).name}
                              </span>
                              <KTooltip wide text={`Cosmic Phase — Where is the network in its deployment lifecycle?\n\nJust as the universe progresses through distinct eras — the radiation era (hot chaos), the matter era (structure forming), the dark energy era (accelerating expansion) — a distributed blockchain network progresses through four phases as servers adopt new software:\n\n1. Isolation — Servers are independent islands, like separate galaxies before first contact. New binaries are being tested in sandboxed environments (Alpha Docker, Delta canary). No production traffic is affected. Risk level: none.\n\n2. Convergence — Servers begin synchronizing, like galaxies merging. Gamma receives the new binary and proves it can reach the same blockchain state as Beta. This is the "peer review" phase. Risk level: low.\n\n3. Aeon Transition — The critical handoff moment, like a star collapsing into a new state. Beta (the production primary) is being upgraded while Gamma temporarily serves all user traffic. Named after Penrose's Conformal Cyclic Cosmology, where the boundary between cosmic epochs is a moment of transformation. Risk level: moderate.\n\n4. Harmony — All servers converged. Same version, same height, same consensus. Like planets in orbital resonance, every node reinforces the others. This is the target state. Risk level: zero.\n\nThe phase automatically advances as servers pass health checks. You cannot skip phases.`}>
                                <span className="text-[9px] text-amber-200/40 cursor-help border-b border-dotted border-amber-200/20">Cosmic Phase</span>
                              </KTooltip>
                            </div>
                            <KTooltip wide text={`K-Kristensen Convergence Readiness — Is it safe to deploy right now?\n\nThis single number (0 to 1) answers the most important question in distributed systems: "If I push new code, will the network break?"\n\nThe K-Parameter is inspired by the Drake Equation from astrobiology. Frank Drake estimated the number of alien civilizations by multiplying independent probability factors: star formation rate × fraction with planets × fraction with life × ... The insight was that one formula could compress enormous complexity into a single meaningful number.\n\nThe K-Parameter does the same thing for network health:\n\nk = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20\n\nFive colored progress bars, five independent health factors:\n  G (green)  = Genetic Stability — version compatibility\n  Q (cyan)   = Quantum Coherence — state synchronization\n  T (orange) = Thermodynamic Efficiency — resource usage\n  I (blue)   = Information Density — data throughput\n  R (purple) = Network Resilience — connection stability\n\nBecause the factors are MULTIPLIED (not averaged), one zero kills the entire score — like a chain breaking at its weakest link. This is intentional: a node that is perfectly efficient but running an incompatible version (G=0) should score zero.\n\nThe exponents (0.25, 0.20, etc.) are weights. G has the highest weight because version mismatch causes the worst failures. I has the lowest because throughput naturally varies with traffic.\n\nScoring: >0.85 = safe to deploy, 0.50-0.85 = proceed with caution, <0.50 = investigate before touching anything.`}>
                              <div className="text-[10px] text-amber-200/50 mt-0.5 cursor-help border-b border-dotted border-amber-200/20 inline-block">
                                K-Kristensen Convergence Readiness
                              </div>
                            </KTooltip>
                          </div>
                        </div>

                        {/* Collective K gauge */}
                        <div className="flex items-center gap-3">
                          <KTooltip wide text={(() => {
                            const activeNodes = convergence.nodes.filter(n => n.k_parameter > 0);
                            const nodeList = activeNodes.map(n => `  ${n.name.replace('Server ', '')}: K = ${n.k_parameter.toFixed(4)}`).join('\n');
                            const product = activeNodes.reduce((acc, n) => acc * n.k_parameter, 1);
                            const geomMean = activeNodes.length > 0 ? Math.pow(product, 1 / activeNodes.length) : 0;
                            const weakest = activeNodes.reduce((min, n) => n.k_parameter < min.k_parameter ? n : min, activeNodes[0]);
                            const strongest = activeNodes.reduce((max, n) => n.k_parameter > max.k_parameter ? n : max, activeNodes[0]);
                            return `Collective K = ${convergence.collective_k.toFixed(4)}\n\nFormula: geometric_mean(K₁, K₂, ..., Kₙ) = (K₁ × K₂ × ... × Kₙ)^(1/n)\n\nActive nodes (${activeNodes.length}):\n${nodeList}\n\nProduct: ${product.toFixed(6)}\nGeometric mean: ${product.toFixed(6)}^(1/${activeNodes.length}) = ${geomMean.toFixed(4)}\n\nStrongest node: ${strongest?.name.replace('Server ', '') || 'N/A'} (K = ${strongest?.k_parameter.toFixed(4) || 'N/A'})\nWeakest node: ${weakest?.name.replace('Server ', '') || 'N/A'} (K = ${weakest?.k_parameter.toFixed(4) || 'N/A'})\n\nWhy geometric mean? Unlike arithmetic mean, the geometric mean heavily penalizes outliers. If 3 nodes have K=0.95 and 1 node has K=0.10, the arithmetic mean would be 0.74 (looks OK), but the geometric mean drops to 0.43 (correctly flagging the weak link). You're only as strong as your weakest server.\n\nThresholds:\n  >0.85: All servers healthy — safe to deploy\n  0.60-0.85: Caution — investigate lagging nodes\n  <0.60: Do not deploy — at least one node is failing`;
                          })()}>
                            <div className="cursor-help">
                              <KGauge value={convergence.collective_k} label="Collective K" size="lg" />
                            </div>
                          </KTooltip>
                          <div className="flex flex-col items-end gap-0.5">
                            <KTooltip wide text={(() => {
                              const o = getOutcomeInfo(convergence.predicted_outcome).name;
                              if (o === 'Communion') return `Communion — The best possible outcome. Like two galaxies merging peacefully into a larger, more beautiful spiral, all servers are in near-perfect agreement. Version match is confirmed, blockchain heights are synchronized, and the synergy bonus means the combined network is actually stronger than the sum of its parts. In game theory, this is a cooperative equilibrium — all players benefit from working together. Safe to deploy without hesitation.`;
                              if (o === 'Observation') return `Observation — A cautious but stable state. Like astronomers studying a distant galaxy before deciding to send a probe, the network is watching and gathering data. The servers are mostly in agreement but there may be minor version differences or slight height gaps. In diplomacy terms, this is "safe limited contact" — proceed carefully but the risk is low. Deployment is possible but double-check the metrics first.`;
                              if (o === 'Competition') return `Competition — A state of resource equilibrium with tension. Like two species competing for the same ecological niche, the servers are functional but not in perfect harmony. There may be version mismatches, height differences, or one server lagging behind. In economics, this is a Nash equilibrium — no single server can improve without the others adjusting. Deployment is risky: you should verify that the lagging server catches up before pushing new code, or you might create a network split.`;
                              if (o === 'Conflict') return `Conflict — A dangerous state where servers disagree on fundamental state. Like tectonic plates building pressure before an earthquake, the network has significant discrepancies. Servers may be on different versions, different chain heights, or producing conflicting blocks. In distributed systems, this is a "split brain" scenario. DO NOT deploy in this state — resolve the conflicts first by checking server logs, resyncing the lagging node, or rolling back a bad deployment.`;
                              if (o === 'Absorption') return `Absorption — The worst outcome. Like a black hole consuming a star, one version of the network is overwhelming the other. This usually means a deployment went wrong and the new version is incompatible with the old. Emergency rollback is required immediately. In biology, this is analogous to a hostile immune response — the body is rejecting the transplant. Stop all deployment activity and restore the previous binary.`;
                              return `Predicted Outcome — Based on the current K-Parameters and convergence metrics, this is the system's forecast of what will happen if you deploy now.`;
                            })()}>
                              <div className={`text-[10px] font-bold cursor-help ${getOutcomeInfo(convergence.predicted_outcome).color}`}>
                                {getOutcomeInfo(convergence.predicted_outcome).name}
                              </div>
                            </KTooltip>
                            <KTooltip text={(() => {
                              const o = getOutcomeInfo(convergence.predicted_outcome).name;
                              if (o === 'Competition') return `Resource Equilibrium — The servers are in a balanced tension state, like two equally matched chess players. Neither side is winning or losing, but the system isn't fully cooperating either. Resources (CPU, memory, bandwidth, block production) are being contested rather than shared. This often happens when one server is on a newer version than another — they can still communicate, but they're not optimally synchronized. Resolve by ensuring all servers run the same version.`;
                              return getOutcomeInfo(convergence.predicted_outcome).desc;
                            })()}>
                              <div className="text-[9px] text-amber-200/40 cursor-help">
                                {getOutcomeInfo(convergence.predicted_outcome).desc}
                              </div>
                            </KTooltip>
                            {convergence.convergence_safe ? (
                              <KTooltip text="Safe to Deploy — All pre-flight checks have passed. The Collective K is above the safety threshold, servers are synchronized, and the predicted outcome is cooperative. Like a green traffic light, this means you can proceed with the rolling deployment. The system has high confidence that pushing new code will not cause a network disruption, consensus failure, or data loss. Go ahead and hit that deploy button.">
                                <div className="flex items-center gap-0.5 mt-0.5 cursor-help">
                                  <CheckCircle className="w-2.5 h-2.5 text-emerald-400" />
                                  <span className="text-[9px] text-emerald-400 font-medium">Safe to deploy</span>
                                </div>
                              </KTooltip>
                            ) : (
                              <KTooltip wide text="Verify First — The system is NOT confident that deploying right now is safe. This is like a yellow traffic light — you CAN proceed, but you should slow down and check your mirrors. Common reasons: a server is lagging behind in block height, version mismatch between nodes, one node has a low K-Parameter score, or the predicted outcome is competitive/conflictual. Before deploying: check that all servers show the same version, verify block heights are within 5 of each other, and ensure no server has a K below 0.50. If in doubt, wait for the metrics to stabilize.">
                                <div className="flex items-center gap-0.5 mt-0.5 cursor-help">
                                  <AlertTriangle className="w-2.5 h-2.5 text-amber-400" />
                                  <span className="text-[9px] text-amber-400 font-medium">Verify first</span>
                                </div>
                              </KTooltip>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Per-Node K-Metrics */}
                      <div className="space-y-1.5 mb-2">
                        {convergence.nodes.filter(n => n.k_parameter > 0).map(node => (
                          <div key={node.name} className="flex items-center gap-2">
                            <KTooltip wide text={(() => {
                              const n = node.name.replace('Server ', '');
                              const k = node.k_parameter;
                              const g = node.genetic_stability;
                              const q = node.quantum_coherence;
                              const t = node.thermodynamic_efficiency;
                              const i = node.information_density;
                              const r = node.network_resilience;
                              const health = k > 0.9 ? 'excellent — performing at peak capacity' : k > 0.7 ? 'good — healthy and contributing to the network' : k > 0.5 ? 'moderate — some factors pulling the score down' : 'poor — needs immediate attention';
                              const weakest = Math.min(g, q, t, i, r);
                              const weakLabel = weakest === g ? 'G (Genetic Stability)' : weakest === q ? 'Q (Quantum Coherence)' : weakest === t ? 'T (Thermodynamic Eff.)' : weakest === i ? 'I (Information Density)' : 'R (Network Resilience)';
                              const strongest = Math.max(g, q, t, i, r);
                              const strongLabel = strongest === g ? 'G (Genetic Stability)' : strongest === q ? 'Q (Quantum Coherence)' : strongest === t ? 'T (Thermodynamic Eff.)' : strongest === i ? 'I (Information Density)' : 'R (Network Resilience)';
                              return `${n} — K = ${k.toFixed(4)} | Health: ${health}\n\nFactor breakdown:\n  G = ${(g*100).toFixed(1)}%  Genetic Stability (version match)\n  Q = ${(q*100).toFixed(1)}%  Quantum Coherence (block sync)\n  T = ${(t*100).toFixed(1)}%  Thermodynamic Efficiency (resources)\n  I = ${(i*100).toFixed(1)}%  Information Density (throughput)\n  R = ${(r*100).toFixed(1)}%  Network Resilience (connectivity)\n\nStrongest: ${strongLabel} at ${(strongest*100).toFixed(1)}%\nWeakest: ${weakLabel} at ${(weakest*100).toFixed(1)}%\n\nThe final K is a weighted geometric mean — a single weak factor drags the entire score down disproportionately. Hover each colored bar for detailed computation.`;
                            })()}>
                              <span className="text-[10px] text-amber-200/60 w-16 truncate cursor-help">{node.name.replace('Server ', '')}</span>
                            </KTooltip>
                            <div className="flex-1">
                              <KMetricsBar metrics={node} />
                            </div>
                            <KTooltip wide text={(() => {
                              const g = node.genetic_stability;
                              const q = node.quantum_coherence;
                              const t = node.thermodynamic_efficiency;
                              const i = node.information_density;
                              const r = node.network_resilience;
                              const gw = Math.pow(g, 0.25);
                              const qw = Math.pow(q, 0.20);
                              const tw = Math.pow(t, 0.20);
                              const iw = Math.pow(i, 0.15);
                              const rw = Math.pow(r, 0.20);
                              const computed = gw * qw * tw * iw * rw;
                              const weakest = Math.min(g, q, t, i, r);
                              const weakLabel = weakest === g ? 'G (Genetic Stability)' : weakest === q ? 'Q (Quantum Coherence)' : weakest === t ? 'T (Thermodynamic Efficiency)' : weakest === i ? 'I (Information Density)' : 'R (Network Resilience)';
                              return `K = ${node.k_parameter.toFixed(4)} — Live Computation Breakdown\n\nFormula: k = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20\n\nStep-by-step with actual values:\n  G = ${g.toFixed(4)}  →  ${g.toFixed(4)}^0.25 = ${gw.toFixed(4)}  (weight: 25%)\n  Q = ${q.toFixed(4)}  →  ${q.toFixed(4)}^0.20 = ${qw.toFixed(4)}  (weight: 20%)\n  T = ${t.toFixed(4)}  →  ${t.toFixed(4)}^0.20 = ${tw.toFixed(4)}  (weight: 20%)\n  I = ${i.toFixed(4)}  →  ${i.toFixed(4)}^0.15 = ${iw.toFixed(4)}  (weight: 15%)\n  R = ${r.toFixed(4)}  →  ${r.toFixed(4)}^0.20 = ${rw.toFixed(4)}  (weight: 20%)\n\nProduct: ${gw.toFixed(4)} × ${qw.toFixed(4)} × ${tw.toFixed(4)} × ${iw.toFixed(4)} × ${rw.toFixed(4)} = ${computed.toFixed(4)}\n\nWeakest link: ${weakLabel} at ${(weakest * 100).toFixed(1)}%\n\nColor coding:\n  Green (>0.90): Excellent — deploy with confidence\n  Blue (>0.70): Good — safe to deploy\n  Amber (>0.50): Caution — review weak factors first\n  Red (<0.50): Critical — do not deploy, investigate`;
                            })()}>
                              <span className={`text-[10px] font-mono font-bold w-8 text-right cursor-help ${
                                node.k_parameter > 0.9 ? 'text-emerald-400' :
                                node.k_parameter > 0.7 ? 'text-blue-400' :
                                node.k_parameter > 0.5 ? 'text-amber-400' : 'text-red-400'
                              }`}>
                                {node.k_parameter.toFixed(2)}
                              </span>
                            </KTooltip>
                          </div>
                        ))}
                      </div>

                      {/* K-Formula legend */}
                      <KTooltip wide text={`The K-Parameter Formula — Inspired by the Drake Equation from astrobiology.\n\nk = G^0.25 × Q^0.20 × T^0.20 × I^0.15 × R^0.20\n\nEach letter represents a measurable property of a network node:\n\n• G (Genetic Stability, weight 25%) — Are all servers running compatible software versions? Like DNA compatibility in biology, mismatched versions cause "genetic" conflicts. This has the highest weight because version mismatch is the #1 cause of deployment failures.\n\n• Q (Quantum Coherence, weight 20%) — How well is the node maintaining consensus with its peers? Named after quantum coherence in physics, where particles maintain correlated states. High Q means the node's blockchain state is identical to the network's.\n\n• T (Thermodynamic Efficiency, weight 20%) — Is the node using its resources (CPU, memory, disk I/O) efficiently? In thermodynamics, efficiency measures useful work vs. wasted energy. A node with high CPU usage but low block production has poor T.\n\n• I (Information Density, weight 15%) — How much useful data is the node processing per unit time? In information theory, density measures signal vs. noise. A node producing many blocks with valid transactions has high I. This has the lowest weight because information throughput varies naturally with network load.\n\n• R (Network Resilience, weight 20%) — Can the node recover from failures and maintain connections? Like ecological resilience, this measures how well the node bounces back from disruptions — network partitions, peer disconnections, or high latency.\n\nThe final K is always between 0 and 1. Because the factors are MULTIPLIED (not averaged), a single zero factor kills the entire score — just like one broken link breaks a chain.`}>
                        <div className="flex items-center justify-center gap-1 text-[10px] text-amber-200/40 mb-2 cursor-help border-b border-dotted border-amber-200/10 pb-1 mx-auto w-fit font-mono">
                          <span className="text-amber-200/60 font-bold italic">k</span>
                          <span>=</span>
                          <span className="text-emerald-400/70">G</span><span className="text-[7px] text-amber-200/30 relative" style={{top: '-4px'}}>.25</span>
                          <span className="text-amber-200/20 mx-0.5">&times;</span>
                          <span className="text-cyan-400/70">Q</span><span className="text-[7px] text-amber-200/30 relative" style={{top: '-4px'}}>.20</span>
                          <span className="text-amber-200/20 mx-0.5">&times;</span>
                          <span className="text-orange-400/70">T</span><span className="text-[7px] text-amber-200/30 relative" style={{top: '-4px'}}>.20</span>
                          <span className="text-amber-200/20 mx-0.5">&times;</span>
                          <span className="text-blue-400/70">I</span><span className="text-[7px] text-amber-200/30 relative" style={{top: '-4px'}}>.15</span>
                          <span className="text-amber-200/20 mx-0.5">&times;</span>
                          <span className="text-purple-400/70">R</span><span className="text-[7px] text-amber-200/30 relative" style={{top: '-4px'}}>.20</span>
                        </div>
                      </KTooltip>

                      {/* Gardener Wisdom */}
                      <div className="rounded-lg bg-slate-800/30 border border-slate-700/20 px-3 py-2">
                        <p className="text-[10px] text-amber-200/60 italic leading-relaxed">
                          "{convergence.gardener_wisdom}"
                        </p>
                        <p className="text-[8px] text-amber-200/30 mt-1 text-right">
                          — The Cosmic Gardener
                          {convergence.phase_transition_eta && (
                            <span className="ml-2">
                              Next phase: ~{convergence.phase_transition_eta < 60
                                ? `${convergence.phase_transition_eta}s`
                                : `${Math.floor(convergence.phase_transition_eta / 60)}m`}
                            </span>
                          )}
                        </p>
                      </div>
                    </motion.div>
                  )}

                  {/* Network Stats Bar */}
                  <div className="grid grid-cols-4 gap-2">
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2 text-center">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Height Delta</div>
                      <div className={`text-sm font-bold ${
                        Math.abs(deployStatus.height_delta) <= 5 ? 'text-emerald-400' :
                        Math.abs(deployStatus.height_delta) <= 100 ? 'text-amber-400' :
                        'text-red-400'
                      }`}>
                        {deployStatus.height_delta > 0 ? '+' : ''}{formatNumber(Math.abs(deployStatus.height_delta))}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2 text-center">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Versions</div>
                      <div className={`text-sm font-bold ${deployStatus.versions_match ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {deployStatus.versions_match ? 'Match' : 'Differ'}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2 text-center">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Total Peers</div>
                      <div className="text-sm font-bold text-blue-400">
                        {(deployStatus.alpha.online ? deployStatus.alpha.peers : 0) + (deployStatus.beta.online ? deployStatus.beta.peers : 0) + (deployStatus.gamma.online ? deployStatus.gamma.peers : 0) + (deployStatus.delta?.online ? deployStatus.delta.peers : 0) + (deployStatus.epsilon?.online ? deployStatus.epsilon.peers : 0)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2 text-center">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Servers</div>
                      <div className="text-sm font-bold text-emerald-400">
                        {(deployStatus.alpha.online ? 1 : 0) + (deployStatus.beta.online ? 1 : 0) + (deployStatus.gamma.online ? 1 : 0) + (deployStatus.delta?.online ? 1 : 0) + (deployStatus.epsilon?.online ? 1 : 0)}/5
                      </div>
                    </div>
                  </div>

                  {/* Refresh info */}
                  <div className="flex items-center justify-end gap-2 text-[10px] text-amber-200/40">
                    {lastRefresh && (
                      <span>Updated {lastRefresh.toLocaleTimeString()}</span>
                    )}
                    <button
                      onClick={fetchStatus}
                      disabled={loading}
                      className="p-1 rounded hover:bg-white/10 transition-colors"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 text-amber-300/60 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                  </div>
                </>
              ) : loading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-emerald-400 animate-spin" />
                </div>
              ) : (
                <div className="text-center py-8 text-amber-200/40 text-sm">
                  No status data available
                </div>
              )}

              {/* Dev Fee Verification & Config */}
              {devFee && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <DollarSign className="w-4 h-4 text-amber-400" />
                      <span className="text-xs font-semibold text-amber-200">Dev Fee Verification</span>
                    </div>
                    <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      devFee.fee_verified
                        ? 'bg-emerald-500/20 border border-emerald-400/40 text-emerald-300'
                        : 'bg-red-500/20 border border-red-400/40 text-red-300'
                    }`}>
                      {devFee.fee_verified ? (
                        <><CheckCircle className="w-2.5 h-2.5" /> VERIFIED</>
                      ) : (
                        <><AlertTriangle className="w-2.5 h-2.5" /> MISMATCH</>
                      )}
                    </div>
                  </div>

                  {/* Fee Stats Grid */}
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Current Fee</div>
                      <div className="text-sm font-bold text-amber-300">{devFee.fee_percent}</div>
                      <div className="text-[10px] text-amber-200/40">{devFee.fee_bps} bps</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Founder Balance</div>
                      <div className="text-sm font-bold text-emerald-400">{devFee.founder_balance_qug.toFixed(4)}</div>
                      <div className="text-[10px] text-amber-200/40">QUG</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Blocks</div>
                      <div className="text-sm font-bold text-blue-400">{formatNumber(devFee.blocks_processed)}</div>
                      <div className="text-[10px] text-amber-200/40">processed</div>
                    </div>
                  </div>

                  {/* Fee Comparison */}
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Total Collected</div>
                      <div className="text-xs font-bold text-amber-300">{devFee.total_dev_fees_collected.toFixed(6)} QUG</div>
                      <div className="text-[10px] text-amber-200/40">
                        Ratio: {(devFee.actual_fee_ratio * 100).toFixed(3)}%
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Total Rewards</div>
                      <div className="text-xs font-bold text-emerald-300">{devFee.total_mining_rewards.toFixed(6)} QUG</div>
                      <div className="text-[10px] text-amber-200/40">
                        Expected: {(devFee.expected_fee_ratio * 100).toFixed(3)}%
                      </div>
                    </div>
                  </div>

                  {/* Today's Stats */}
                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <div className="rounded-lg bg-slate-800/30 border border-slate-700/30 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Today Dev Fee</div>
                      <div className="text-xs font-bold text-amber-300">{devFee.today_dev_fee_qug.toFixed(6)} QUG</div>
                    </div>
                    <div className="rounded-lg bg-slate-800/30 border border-slate-700/30 p-2">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Today Expected</div>
                      <div className="text-xs font-bold text-emerald-300">{devFee.today_expected_dev_fee_qug.toFixed(6)} QUG</div>
                    </div>
                  </div>

                  {/* Fee Config */}
                  <div className="rounded-lg bg-slate-800/30 border border-slate-700/30 p-2">
                    <div className="flex items-center gap-2 mb-2">
                      <Settings className="w-3 h-3 text-amber-400" />
                      <span className="text-[10px] font-semibold text-amber-200/70 uppercase tracking-wider">Adjust Dev Fee</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="0"
                        max="1000"
                        value={devFeeInput}
                        onChange={(e) => setDevFeeInput(e.target.value)}
                        className="flex-1 bg-slate-900/50 border border-slate-600/50 rounded-lg px-3 py-1.5 text-xs text-amber-50 focus:border-amber-400/50 focus:outline-none"
                        placeholder="100"
                      />
                      <span className="text-[10px] text-amber-200/50 w-12">
                        = {devFeeInput ? (parseInt(devFeeInput) / 100).toFixed(2) : '?'}%
                      </span>
                      <motion.button
                        onClick={saveDevFee}
                        disabled={devFeeSaving}
                        className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all disabled:opacity-50"
                        style={{
                          background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%)',
                          border: '1px solid rgba(245, 158, 11, 0.4)',
                          color: 'rgb(253, 230, 138)',
                        }}
                        whileHover={{ scale: devFeeSaving ? 1 : 1.03 }}
                        whileTap={{ scale: devFeeSaving ? 1 : 0.97 }}
                      >
                        {devFeeSaving ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                        Save
                      </motion.button>
                    </div>
                    {devFeeMsg && (
                      <div className={`mt-2 text-[10px] ${devFeeMsg.type === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
                        {devFeeMsg.text}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Verification Progress */}
              {verifyEvents.length > 0 && (
                <div className="rounded-xl border border-slate-700/50 bg-slate-900/30 overflow-hidden">
                  <div className="px-4 py-2.5 border-b border-slate-700/30 bg-slate-800/30">
                    <div className="flex items-center gap-2">
                      <Activity className={`w-4 h-4 ${(isVerifying || pipelineRunning) ? 'text-blue-400 animate-pulse' : allPassed ? 'text-emerald-400' : 'text-red-400'}`} />
                      <span className="text-sm font-medium text-amber-50">
                        {pipelineRunning ? 'Pipeline Running...' :
                         isVerifying ? 'Verification In Progress...' :
                         allPassed ? 'PASSED' : anyFailed ? 'FAILED' : 'Complete'}
                      </span>
                    </div>
                  </div>
                  <div className="p-3 space-y-1.5 max-h-48 overflow-y-auto">
                    {verifyEvents.map((event, i) => {
                      // Step-specific colors
                      const stepColor = event.step === 'alpha-canary' ? 'text-purple-400' :
                        event.step === 'gamma-verify' ? 'text-blue-400' :
                        event.step === 'soak-test' ? 'text-amber-400' :
                        event.step === 'beta-deploy' ? 'text-emerald-400' :
                        event.step === 'auto-rollback' ? 'text-red-400' :
                        'text-amber-200/80';
                      return (
                        <div key={i} className="flex items-start gap-2 text-xs">
                          {event.status === 'passed' ? (
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                          ) : event.status === 'failed' ? (
                            <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                          ) : event.step === 'soak-test' ? (
                            <Timer className="w-3.5 h-3.5 text-amber-400 animate-pulse flex-shrink-0 mt-0.5" />
                          ) : (
                            <RefreshCw className="w-3.5 h-3.5 text-blue-400 animate-spin flex-shrink-0 mt-0.5" />
                          )}
                          <div>
                            <span className={`font-medium ${stepColor}`}>{event.step}: </span>
                            <span className="text-amber-200/70">{event.message}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Deploy Pipeline Visual — CCC Phase Mapping */}
              <div className="rounded-xl border border-slate-700/30 bg-slate-900/20 p-3">
                <div className="flex items-center gap-1 mb-2">
                  <Database className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-[10px] font-semibold text-amber-200/70 uppercase tracking-wider">
                    Cosmic Deploy Pipeline
                  </span>
                </div>
                <div className="flex items-center justify-center gap-1 text-[10px]">
                  {/* Parallel: Alpha + Delta */}
                  <div className="flex flex-col items-center gap-1 rounded-lg border border-dashed border-slate-600/40 px-2 py-1">
                    <span className="text-[7px] text-amber-200/40 uppercase tracking-wider">parallel</span>
                    <div className="flex items-center gap-1">
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="px-1.5 py-0.5 rounded bg-purple-500/20 border border-purple-400/30 text-purple-300 font-medium text-[10px]">
                          Alpha
                        </span>
                        <span className="text-[7px] text-purple-400/60">Canary</span>
                      </div>
                      <span className="text-[8px] text-amber-200/30">+</span>
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 border border-cyan-400/30 text-cyan-300 font-medium text-[10px]">
                          Delta
                        </span>
                        <span className="text-[7px] text-cyan-400/60">Bootstrap</span>
                      </div>
                    </div>
                  </div>
                  <ArrowRight className="w-3 h-3 text-amber-200/30" />
                  {/* Convergence → Gamma */}
                  <div className="flex flex-col items-center gap-0.5">
                    <span className="px-2 py-0.5 rounded bg-blue-500/20 border border-blue-400/30 text-blue-300 font-medium">
                      Gamma
                    </span>
                    <span className="text-[8px] text-blue-400/60">Verify</span>
                  </div>
                  <ArrowRight className="w-3 h-3 text-amber-200/30" />
                  {/* Aeon Transition → Beta */}
                  <div className="flex flex-col items-center gap-0.5">
                    <span className="px-2 py-0.5 rounded bg-amber-500/20 border border-amber-400/30 text-amber-300 font-medium">
                      Beta
                    </span>
                    <span className="text-[8px] text-amber-400/60">Primary</span>
                  </div>
                  <ArrowRight className="w-3 h-3 text-amber-200/30" />
                  {/* Harmony */}
                  <div className="flex flex-col items-center gap-0.5">
                    <span className="px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 font-medium">
                      Harmony
                    </span>
                    <span className="text-[8px] text-emerald-400/60">k &gt; 0.9</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons — v8.6.4: Only visible for master/admin wallet */}
              {isMaster && (<>
              <div className="flex gap-2">
                <motion.button
                  onClick={startVerification}
                  disabled={isVerifying}
                  className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl font-medium text-xs transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%)',
                    border: '1px solid rgba(59, 130, 246, 0.4)',
                    color: 'rgb(147, 197, 253)',
                  }}
                  whileHover={{ scale: isVerifying ? 1 : 1.02 }}
                  whileTap={{ scale: isVerifying ? 1 : 0.98 }}
                >
                  {isVerifying ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Activity className="w-3.5 h-3.5" />
                  )}
                  {isVerifying ? 'Verifying...' : 'Verify'}
                </motion.button>

                <motion.button
                  onClick={triggerDeployAll}
                  disabled={isVerifying || pipelineRunning}
                  className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl font-medium text-xs transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.2) 100%)',
                    border: '1px solid rgba(16, 185, 129, 0.4)',
                    color: 'rgb(167, 243, 208)',
                  }}
                  whileHover={{ scale: (isVerifying || pipelineRunning) ? 1 : 1.02 }}
                  whileTap={{ scale: (isVerifying || pipelineRunning) ? 1 : 0.98 }}
                >
                  {pipelineRunning ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Rocket className="w-3.5 h-3.5" />
                  )}
                  {pipelineRunning ? 'Deploying...' : 'Deploy All'}
                </motion.button>
              </div>

              {/* Rollback */}
              <motion.button
                onClick={triggerRollback}
                disabled={pipelineRunning}
                className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-xl text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: 'rgba(239, 68, 68, 0.08)',
                  border: '1px solid rgba(239, 68, 68, 0.25)',
                  color: 'rgb(252, 165, 165)',
                }}
                whileHover={{ scale: pipelineRunning ? 1 : 1.01 }}
                whileTap={{ scale: pipelineRunning ? 1 : 0.99 }}
              >
                <RotateCcw className="w-4 h-4" />
                Rollback
              </motion.button>
              </>)}
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* ANALYTICS TAB — Live sparkline charts                      */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'analytics' && (<>
                {/* Header */}
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold text-cyan-200/80 uppercase tracking-wider">Live Metrics</span>
                    <span className="text-[9px] text-slate-500">
                      {metricsHistory.length} samples ({metricsHistory.length > 0 ? `${Math.round((Date.now() - metricsHistory[0].ts) / 60000)}m window` : 'collecting...'})
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-[9px] text-emerald-300/70">Polling 15s</span>
                  </div>
                </div>

                {/* q-flux Section */}
                <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Zap className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-[11px] font-semibold text-cyan-300">q-flux Reverse Proxy</span>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <AnalyticsCard
                      title="Requests/s"
                      value={metricsHistory.length > 0 ? (metricsHistory[metricsHistory.length - 1].flux_rps > 0 ? metricsHistory[metricsHistory.length - 1].flux_rps.toFixed(0) : '0') : '-'}
                      unit="req/s"
                      data={metricsHistory.map(s => s.flux_rps)}
                      color="#06b6d4"
                      tooltip="HTTP requests per second handled by q-flux. Each sample is from the 15-second polling interval."
                    />
                    <AnalyticsCard
                      title="Error Rate"
                      value={metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1].flux_err_pct.toFixed(2) : '-'}
                      unit="%"
                      data={metricsHistory.map(s => s.flux_err_pct)}
                      color={metricsHistory.length > 0 && metricsHistory[metricsHistory.length - 1].flux_err_pct > 1 ? '#ef4444' : '#10b981'}
                      tooltip="Percentage of responses with 5xx status codes. Green <1%, red >1%."
                      format={(v: number) => v.toFixed(2)}
                    />
                    <AnalyticsCard
                      title="Active Connections"
                      value={metricsHistory.length > 0 ? formatNumber(metricsHistory[metricsHistory.length - 1].flux_active_conns) : '-'}
                      data={metricsHistory.map(s => s.flux_active_conns)}
                      color="#8b5cf6"
                      tooltip="Currently open TCP connections. Includes miners, wallets, SSE streams."
                    />
                    <AnalyticsCard
                      title="Upstream Active"
                      value={metricsHistory.length > 0 ? String(metricsHistory[metricsHistory.length - 1].flux_upstream_active) : '-'}
                      data={metricsHistory.map(s => s.flux_upstream_active)}
                      color="#f59e0b"
                      tooltip="Concurrent requests forwarded to the backend. Global semaphore capped at 1024. If this hits the cap, new requests are rejected (503)."
                    />
                    <AnalyticsCard
                      title="WebSockets"
                      value={metricsHistory.length > 0 ? String(metricsHistory[metricsHistory.length - 1].flux_active_ws) : '-'}
                      data={metricsHistory.map(s => s.flux_active_ws)}
                      color="#a855f7"
                      tooltip="Active WebSocket connections. Each connected wallet or miner may hold one persistent WS."
                    />
                    <AnalyticsCard
                      title="H2 Streams"
                      value={metricsHistory.length > 0 ? formatNumber(metricsHistory[metricsHistory.length - 1].flux_h2_active) : '-'}
                      data={metricsHistory.map(s => s.flux_h2_active)}
                      color="#10b981"
                      tooltip="Active HTTP/2 multiplexed streams (opened minus closed)."
                    />
                  </div>

                  {/* Bandwidth sparklines (special: show delta bytes/s) */}
                  {metricsHistory.length >= 2 && (() => {
                    // Compute bytes/sec deltas between consecutive samples
                    const rxRate: number[] = [];
                    const txRate: number[] = [];
                    for (let i = 1; i < metricsHistory.length; i++) {
                      const dt = (metricsHistory[i].ts - metricsHistory[i - 1].ts) / 1000;
                      if (dt > 0) {
                        rxRate.push(Math.max(0, (metricsHistory[i].flux_bytes_rx - metricsHistory[i - 1].flux_bytes_rx) / dt));
                        txRate.push(Math.max(0, (metricsHistory[i].flux_bytes_tx - metricsHistory[i - 1].flux_bytes_tx) / dt));
                      }
                    }
                    const fmtBw = (v: number) => v >= 1073741824 ? `${(v / 1073741824).toFixed(1)}GB/s` : v >= 1048576 ? `${(v / 1048576).toFixed(1)}MB/s` : v >= 1024 ? `${(v / 1024).toFixed(1)}KB/s` : `${v.toFixed(0)}B/s`;
                    const lastRx = rxRate.length > 0 ? rxRate[rxRate.length - 1] : 0;
                    const lastTx = txRate.length > 0 ? txRate[txRate.length - 1] : 0;
                    return (
                      <div className="grid grid-cols-2 gap-3 pt-1 border-t border-cyan-500/10">
                        <AnalyticsCard
                          title="Bandwidth RX"
                          value={fmtBw(lastRx)}
                          data={rxRate}
                          color="#3b82f6"
                          tooltip="Inbound bandwidth (bytes received from clients per second)."
                          format={fmtBw}
                        />
                        <AnalyticsCard
                          title="Bandwidth TX"
                          value={fmtBw(lastTx)}
                          data={txRate}
                          color="#10b981"
                          tooltip="Outbound bandwidth (bytes sent to clients per second)."
                          format={fmtBw}
                        />
                      </div>
                    );
                  })()}
                </div>

                {/* Caddy Section */}
                <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3 space-y-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Globe className="w-3.5 h-3.5 text-blue-400" />
                    <span className="text-[11px] font-semibold text-blue-300">Caddy Reverse Proxy (Epsilon)</span>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <AnalyticsCard
                      title="Requests/s"
                      value={metricsHistory.length > 0 ? (metricsHistory[metricsHistory.length - 1].caddy_rps > 0 ? metricsHistory[metricsHistory.length - 1].caddy_rps.toFixed(0) : '0') : '-'}
                      unit="req/s"
                      data={metricsHistory.map(s => s.caddy_rps)}
                      color="#3b82f6"
                      tooltip="HTTP requests per second through Caddy on Epsilon."
                    />
                    <AnalyticsCard
                      title="Avg Latency"
                      value={metricsHistory.length > 0 ? (metricsHistory[metricsHistory.length - 1].caddy_avg_ms < 1 ? '<1' : metricsHistory[metricsHistory.length - 1].caddy_avg_ms.toFixed(0)) : '-'}
                      unit="ms"
                      data={metricsHistory.map(s => s.caddy_avg_ms)}
                      color={metricsHistory.length > 0 && metricsHistory[metricsHistory.length - 1].caddy_avg_ms > 100 ? '#f59e0b' : '#10b981'}
                      tooltip="Average response time in milliseconds. Green <100ms, amber >100ms."
                      format={(v: number) => v < 1 ? '<1' : v.toFixed(0)}
                    />
                    <AnalyticsCard
                      title="p99 Latency"
                      value={metricsHistory.length > 0 ? (metricsHistory[metricsHistory.length - 1].caddy_p99_ms >= 1000 ? `${(metricsHistory[metricsHistory.length - 1].caddy_p99_ms / 1000).toFixed(1)}s` : metricsHistory[metricsHistory.length - 1].caddy_p99_ms.toFixed(0)) : '-'}
                      unit="ms"
                      data={metricsHistory.map(s => s.caddy_p99_ms)}
                      color={metricsHistory.length > 0 && metricsHistory[metricsHistory.length - 1].caddy_p99_ms > 500 ? '#ef4444' : '#10b981'}
                      tooltip="99th percentile response time. The slowest 1% of requests."
                      format={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}s` : v.toFixed(0)}
                    />
                    <AnalyticsCard
                      title="Error Rate"
                      value={metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1].caddy_err_pct.toFixed(2) : '-'}
                      unit="%"
                      data={metricsHistory.map(s => s.caddy_err_pct)}
                      color={metricsHistory.length > 0 && metricsHistory[metricsHistory.length - 1].caddy_err_pct > 1 ? '#ef4444' : '#10b981'}
                      tooltip="Caddy 5xx error rate."
                      format={(v: number) => v.toFixed(2)}
                    />
                    <AnalyticsCard
                      title="Goroutines"
                      value={metricsHistory.length > 0 ? (metricsHistory[metricsHistory.length - 1].caddy_goroutines > 1000 ? `${(metricsHistory[metricsHistory.length - 1].caddy_goroutines / 1000).toFixed(1)}K` : String(metricsHistory[metricsHistory.length - 1].caddy_goroutines)) : '-'}
                      data={metricsHistory.map(s => s.caddy_goroutines)}
                      color="#6366f1"
                      tooltip="Caddy goroutines (concurrent tasks). Rising steadily = possible connection leak."
                    />
                    <AnalyticsCard
                      title="Heap Memory"
                      value={metricsHistory.length > 0 ? `${metricsHistory[metricsHistory.length - 1].caddy_memory_mb.toFixed(0)}` : '-'}
                      unit="MB"
                      data={metricsHistory.map(s => s.caddy_memory_mb)}
                      color="#f97316"
                      tooltip="Caddy Go runtime heap memory usage."
                      format={(v: number) => v.toFixed(0)}
                    />
                  </div>
                </div>

                {/* Info footer */}
                <div className="text-[9px] text-slate-500 text-center">
                  Charts show rolling {MAX_HISTORY}-sample window (up to {Math.round(MAX_HISTORY * 15 / 60)} minutes at 15s intervals). Data is in-memory only and resets on page reload.
                </div>
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* BANK CLI TAB                                               */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'bank' && (<>
                {/* Bank CLI Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-amber-400" />
                    <span className="text-xs font-bold text-amber-200/80 uppercase tracking-wider">
                      Quillon Bank Administration
                    </span>
                  </div>
                  <motion.button
                    onClick={fetchBankData}
                    disabled={bankLoading}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] text-emerald-300/70 hover:text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 transition-all"
                  >
                    <RefreshCw className={`w-3 h-3 ${bankLoading ? 'animate-spin' : ''}`} />
                    Refresh
                  </motion.button>
                </div>

                {/* Bank Nav Sections */}
                <div className="flex flex-wrap gap-1">
                  {([
                    { id: 'dashboard', icon: Eye, label: 'Dashboard' },
                    { id: 'mint', icon: Banknote, label: 'Mint/Burn' },
                    { id: 'loans', icon: FileText, label: 'Loans' },
                    { id: 'messages', icon: Send, label: 'Messages' },
                    { id: 'treasury', icon: Wallet, label: 'Treasury' },
                    { id: 'log', icon: Terminal, label: 'Command Log' },
                  ] as const).map(s => (
                    <button
                      key={s.id}
                      onClick={() => setBankSection(s.id)}
                      className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all ${
                        bankSection === s.id
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-400/40'
                          : 'text-amber-200/40 hover:text-amber-200/70 hover:bg-slate-800/40 border border-transparent'
                      }`}
                    >
                      <s.icon className="w-3 h-3" />
                      {s.label}
                      {s.id === 'loans' && bankLoans.filter(l => l.status === 'pending').length > 0 && (
                        <span className="ml-0.5 px-1 rounded-full bg-red-500/30 text-red-300 text-[8px] font-bold">
                          {bankLoans.filter(l => l.status === 'pending').length}
                        </span>
                      )}
                      {s.id === 'messages' && bankMessages.filter((m: any) => !m.read && m.from !== 'Bank').length > 0 && (
                        <span className="ml-0.5 px-1 rounded-full bg-red-500/30 text-red-300 text-[8px] font-bold">
                          {bankMessages.filter((m: any) => !m.read && m.from !== 'Bank').length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                {/* ─── Dashboard Section ─── */}
                {bankSection === 'dashboard' && (
                  <div className="space-y-3">
                    {bankMetrics ? (
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          { label: 'Total Accounts', value: bankMetrics.total_accounts ?? 0, icon: Users, color: 'text-blue-400' },
                          { label: 'Active Loans', value: bankMetrics.active_loans ?? bankLoans.filter(l => l.status === 'approved').length, icon: FileText, color: 'text-amber-400' },
                          { label: 'Pending Loans', value: bankLoans.filter(l => l.status === 'pending').length, icon: Clock, color: 'text-purple-400' },
                          { label: 'At-Risk Loans', value: bankAtRisk?.length ?? 0, icon: AlertTriangle, color: 'text-red-400' },
                          { label: 'Total Deposits', value: `${((bankMetrics.total_deposits ?? 0) / 1e24).toFixed(2)} QUG`, icon: DollarSign, color: 'text-emerald-400' },
                          { label: 'Messages', value: bankMessages?.length ?? 0, icon: Send, color: 'text-indigo-400' },
                        ].map(stat => (
                          <div key={stat.label} className="rounded-lg border border-slate-700/30 bg-slate-800/30 p-2.5">
                            <div className="flex items-center gap-1.5 mb-1">
                              <stat.icon className={`w-3 h-3 ${stat.color}`} />
                              <span className="text-[9px] text-amber-200/50 uppercase tracking-wider">{stat.label}</span>
                            </div>
                            <span className="text-sm font-bold text-amber-50">
                              {typeof stat.value === 'number' ? stat.value.toLocaleString() : stat.value}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 text-amber-200/40 text-xs">
                        {bankLoading ? (
                          <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-emerald-400/50" />
                        ) : (
                          <Landmark className="w-5 h-5 mx-auto mb-2 text-amber-200/30" />
                        )}
                        {bankLoading ? 'Loading bank data...' : 'Click Refresh to load bank data'}
                      </div>
                    )}

                    {/* Stablecoin Quick Stats */}
                    <div className="rounded-lg border border-slate-700/30 bg-slate-800/20 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <CreditCard className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-[10px] font-semibold text-emerald-200/80 uppercase">Quick Actions</span>
                      </div>
                      <div className="grid grid-cols-4 gap-1.5">
                        {[
                          { label: 'Stablecoin Status', cmd: 'GET stablecoin/status', path: '/api/v1/quillon-bank/stablecoin/status' },
                          { label: 'Risk Assessment', cmd: 'GET risk/status', path: '/api/v1/quillon-bank/risk/status' },
                          { label: 'Peg Status', cmd: 'GET stablecoin/peg', path: '/api/v1/quillon-bank/stablecoin/peg' },
                          { label: 'Daily Summary', cmd: 'GET analytics/daily-summary', path: '/api/v1/quillon-bank/analytics/daily-summary' },
                        ].map(action => (
                          <motion.button
                            key={action.label}
                            onClick={() => { setBankSection('log'); bankAction(action.cmd, 'GET', action.path); }}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="p-2 rounded-lg text-[10px] text-amber-200/60 hover:text-amber-200 bg-slate-700/20 hover:bg-slate-700/40 border border-slate-600/20 hover:border-amber-400/30 transition-all text-center"
                          >
                            {action.label}
                          </motion.button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* ─── Mint/Burn Section ─── */}
                {bankSection === 'mint' && (
                  <div className="space-y-3">
                    <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                      <div className="flex items-center gap-1.5 mb-3">
                        <Banknote className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs font-semibold text-emerald-200">Mint QUGUSD</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 mb-3">
                        <div>
                          <label className="text-[9px] text-amber-200/50 uppercase tracking-wider block mb-1">Amount (QUGUSD)</label>
                          <input
                            type="number"
                            value={mintAmount}
                            onChange={e => setMintAmount(e.target.value)}
                            placeholder="1000"
                            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-800/60 border border-slate-600/30 text-xs text-amber-50 placeholder-amber-200/20 focus:border-emerald-500/50 focus:outline-none transition-colors"
                          />
                        </div>
                        <div>
                          <label className="text-[9px] text-amber-200/50 uppercase tracking-wider block mb-1">Collateral (QUG)</label>
                          <input
                            type="number"
                            value={mintCollateral}
                            onChange={e => setMintCollateral(e.target.value)}
                            placeholder="25"
                            className="w-full px-2.5 py-1.5 rounded-lg bg-slate-800/60 border border-slate-600/30 text-xs text-amber-50 placeholder-amber-200/20 focus:border-emerald-500/50 focus:outline-none transition-colors"
                          />
                        </div>
                      </div>
                      <div className="mb-3">
                        <label className="text-[9px] text-amber-200/50 uppercase tracking-wider block mb-1">Recipient Wallet (optional, default: founder)</label>
                        <input
                          type="text"
                          value={mintWallet}
                          onChange={e => setMintWallet(e.target.value)}
                          placeholder="qnk..."
                          className="w-full px-2.5 py-1.5 rounded-lg bg-slate-800/60 border border-slate-600/30 text-xs text-amber-50 placeholder-amber-200/20 focus:border-emerald-500/50 focus:outline-none transition-colors font-mono"
                        />
                      </div>
                      <div className="flex gap-2">
                        <motion.button
                          onClick={() => {
                            if (!mintAmount) return;
                            const amt = Math.round(parseFloat(mintAmount) * 1e24);
                            const body: any = {
                              amount: amt,
                              collateral_type: 'QUG',
                              collateral_amount: parseFloat(mintCollateral) || parseFloat(mintAmount) / 3000 * 1.5,
                            };
                            if (mintWallet) body.wallet_address = mintWallet;
                            setBankSection('log');
                            bankAction(
                              `MINT ${mintAmount} QUGUSD (collateral: ${body.collateral_amount.toFixed(2)} QUG)`,
                              'POST',
                              '/api/v1/quillon-bank/stablecoin/mint',
                              body,
                            );
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition-all"
                        >
                          <Banknote className="w-3.5 h-3.5" />
                          Mint QUGUSD
                        </motion.button>
                        <motion.button
                          onClick={() => {
                            if (!mintAmount) return;
                            const amt = Math.round(parseFloat(mintAmount) * 1e24);
                            setBankSection('log');
                            bankAction(
                              `BURN ${mintAmount} QUGUSD`,
                              'POST',
                              '/api/v1/quillon-bank/stablecoin/burn',
                              { amount: amt, recipient: walletAddress, collateral_type: 'QUG' },
                            );
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium bg-red-500/20 text-red-300 border border-red-500/40 hover:bg-red-500/30 transition-all"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          Burn QUGUSD
                        </motion.button>
                      </div>
                    </div>

                    {/* Add Collateral */}
                    <div className="rounded-lg border border-slate-700/30 bg-slate-800/20 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <Database className="w-3.5 h-3.5 text-blue-400" />
                        <span className="text-[10px] font-semibold text-blue-200/80 uppercase">Collateral Management</span>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5">
                        <motion.button
                          onClick={() => { setBankSection('log'); bankAction('GET stablecoin/collateral', 'GET', '/api/v1/quillon-bank/stablecoin/collateral'); }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="p-2 rounded-lg text-[10px] text-blue-200/60 hover:text-blue-200 bg-slate-700/20 hover:bg-slate-700/40 border border-slate-600/20 hover:border-blue-400/30 transition-all"
                        >
                          View Collateral
                        </motion.button>
                        <motion.button
                          onClick={() => {
                            const amt = prompt('QUG amount to add as collateral:');
                            if (!amt) return;
                            setBankSection('log');
                            bankAction(
                              `ADD COLLATERAL ${amt} QUG`,
                              'POST',
                              '/api/v1/quillon-bank/stablecoin/collateral/add',
                              { collateral_type: 'QUG', amount: parseFloat(amt), reason: 'Admin deposit' },
                            );
                          }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="p-2 rounded-lg text-[10px] text-emerald-200/60 hover:text-emerald-200 bg-slate-700/20 hover:bg-slate-700/40 border border-slate-600/20 hover:border-emerald-400/30 transition-all"
                        >
                          + Add Collateral
                        </motion.button>
                      </div>
                    </div>
                  </div>
                )}

                {/* ─── Loans Section ─── */}
                {bankSection === 'loans' && (
                  <div className="space-y-3">
                    {/* Pending Loans */}
                    <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <FileText className="w-3.5 h-3.5 text-amber-400" />
                        <span className="text-[10px] font-semibold text-amber-200/80 uppercase">Pending Loan Applications</span>
                        <span className="ml-auto text-[9px] text-amber-200/40">{bankLoans.filter(l => l.status === 'pending').length} pending</span>
                      </div>

                      {bankLoans.filter(l => l.status === 'pending').length === 0 ? (
                        <div className="text-center py-3 text-[10px] text-amber-200/30">No pending loan applications</div>
                      ) : (
                        <div className="space-y-1.5 max-h-48 overflow-y-auto">
                          {bankLoans.filter(l => l.status === 'pending').map((loan: any) => (
                            <div key={loan.loan_id} className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/40 border border-slate-700/30">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-1.5">
                                  <span className="text-[10px] font-mono text-amber-200/60 truncate">{loan.borrower_address?.slice(0, 20)}...</span>
                                  <span className="px-1 py-0.5 rounded bg-amber-500/20 text-[8px] text-amber-300 font-bold">PENDING</span>
                                </div>
                                <div className="text-[9px] text-amber-200/40 mt-0.5">
                                  {((loan.loan_amount || 0) / 1e24).toFixed(2)} QUGUSD | {loan.collateral_amount?.toFixed(2)} {loan.collateral_type} collateral | {loan.term_months}mo @ {loan.interest_rate?.toFixed(1)}%
                                </div>
                              </div>
                              <motion.button
                                onClick={() => {
                                  setBankSection('log');
                                  bankAction(
                                    `APPROVE LOAN ${loan.loan_id?.slice(0, 8)}`,
                                    'POST',
                                    '/api/v1/quillon-bank/lending/approve',
                                    { loan_id: loan.loan_id },
                                  );
                                }}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                className="px-2 py-1 rounded-lg text-[10px] font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition-all whitespace-nowrap"
                              >
                                <BadgeCheck className="w-3 h-3 inline mr-0.5" />
                                Approve
                              </motion.button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* At-Risk Loans */}
                    {bankAtRisk.length > 0 && (
                      <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                        <div className="flex items-center gap-1.5 mb-2">
                          <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                          <span className="text-[10px] font-semibold text-red-200/80 uppercase">At-Risk Loans (below 120%)</span>
                        </div>
                        <div className="space-y-1.5 max-h-36 overflow-y-auto">
                          {bankAtRisk.map((loan: any, i: number) => (
                            <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/40 border border-red-500/20">
                              <div className="flex-1 min-w-0">
                                <div className="text-[10px] font-mono text-red-200/60 truncate">{loan.loan_id?.slice(0, 16) || loan.borrower_address?.slice(0, 20)}...</div>
                                <div className="text-[9px] text-red-200/40">
                                  Ratio: {loan.collateral_ratio?.toFixed(0)}% | {((loan.loan_amount || 0) / 1e24).toFixed(2)} QUGUSD
                                </div>
                              </div>
                              <motion.button
                                onClick={() => {
                                  if (!confirm(`Liquidate loan ${loan.loan_id?.slice(0, 8)}? This seizes collateral.`)) return;
                                  setBankSection('log');
                                  bankAction(
                                    `LIQUIDATE ${loan.loan_id?.slice(0, 8)}`,
                                    'POST',
                                    '/api/v1/quillon-bank/lending/liquidate',
                                    { loan_id: loan.loan_id },
                                  );
                                }}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                className="px-2 py-1 rounded-lg text-[10px] font-medium bg-red-500/20 text-red-300 border border-red-500/40 hover:bg-red-500/30 transition-all whitespace-nowrap"
                              >
                                Liquidate
                              </motion.button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* All Loans Summary */}
                    <div className="rounded-lg border border-slate-700/30 bg-slate-800/20 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <Layers className="w-3.5 h-3.5 text-blue-400" />
                        <span className="text-[10px] font-semibold text-blue-200/80 uppercase">All Loans ({bankLoans.length})</span>
                      </div>
                      {bankLoans.length === 0 ? (
                        <div className="text-center py-2 text-[10px] text-amber-200/30">No loans found</div>
                      ) : (
                        <div className="space-y-1 max-h-48 overflow-y-auto">
                          {bankLoans.map((loan: any) => {
                            const statusColor = loan.status === 'approved' ? 'text-emerald-300 bg-emerald-500/20' : loan.status === 'pending' ? 'text-amber-300 bg-amber-500/20' : loan.status === 'paid' ? 'text-blue-300 bg-blue-500/20' : 'text-red-300 bg-red-500/20';
                            return (
                              <div key={loan.loan_id} className="flex items-center gap-2 p-1.5 rounded bg-slate-800/30 border border-slate-700/20 text-[10px]">
                                <span className={`px-1 py-0.5 rounded ${statusColor} text-[8px] font-bold uppercase`}>{loan.status}</span>
                                <span className="font-mono text-amber-200/50 truncate flex-1">{loan.borrower_address?.slice(0, 16)}...</span>
                                <span className="text-amber-200/60">{((loan.loan_amount || 0) / 1e24).toFixed(2)} QUGUSD</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ─── Messages Section ─── */}
                {bankSection === 'messages' && (
                  <div className="space-y-3">
                    {/* Unread Messages */}
                    <div className="rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <Send className="w-3.5 h-3.5 text-indigo-400" />
                        <span className="text-[10px] font-semibold text-indigo-200/80 uppercase">User Messages</span>
                        <span className="ml-auto text-[9px] text-amber-200/40">{bankMessages.length} total</span>
                      </div>

                      {bankMessages.length === 0 ? (
                        <div className="text-center py-3 text-[10px] text-amber-200/30">No messages</div>
                      ) : (
                        <div className="space-y-1.5 max-h-64 overflow-y-auto">
                          {bankMessages.slice(0, 20).map((msg: any) => (
                            <div key={msg.id} className={`p-2 rounded-lg border ${
                              msg.from === 'Bank' || msg.from?.Bank !== undefined
                                ? 'bg-emerald-500/5 border-emerald-500/20'
                                : msg.read ? 'bg-slate-800/30 border-slate-700/20' : 'bg-indigo-500/10 border-indigo-400/30'
                            }`}>
                              <div className="flex items-center gap-1.5 mb-1">
                                <span className={`text-[9px] font-bold uppercase ${
                                  msg.from === 'Bank' || msg.from?.Bank !== undefined ? 'text-emerald-300' : 'text-indigo-300'
                                }`}>
                                  {msg.from === 'Bank' || msg.from?.Bank !== undefined ? 'Bank' : 'User'}
                                </span>
                                {msg.subject && <span className="text-[9px] text-amber-200/50">- {msg.subject}</span>}
                                <span className="ml-auto text-[8px] text-amber-200/30 font-mono">
                                  {msg.wallet_address?.slice(0, 12)}...
                                </span>
                                {!msg.read && msg.from !== 'Bank' && msg.from?.Bank === undefined && (
                                  <span className="px-1 rounded bg-indigo-500/30 text-[7px] text-indigo-300 font-bold">NEW</span>
                                )}
                              </div>
                              <div className="text-[10px] text-amber-200/60 line-clamp-2">{msg.content}</div>
                              {msg.from !== 'Bank' && msg.from?.Bank === undefined && (
                                <div className="flex gap-1 mt-1.5">
                                  <motion.button
                                    onClick={() => { setRespondMsgId(msg.id); setRespondContent(''); }}
                                    whileHover={{ scale: 1.05 }}
                                    whileTap={{ scale: 0.95 }}
                                    className="px-1.5 py-0.5 rounded text-[9px] text-indigo-300 bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-500/30 transition-all"
                                  >
                                    Reply
                                  </motion.button>
                                </div>
                              )}
                              {respondMsgId === msg.id && (
                                <div className="mt-2 space-y-1.5">
                                  <textarea
                                    value={respondContent}
                                    onChange={e => setRespondContent(e.target.value)}
                                    placeholder="Type your response..."
                                    rows={2}
                                    className="w-full px-2 py-1.5 rounded-lg bg-slate-800/60 border border-slate-600/30 text-[10px] text-amber-50 placeholder-amber-200/20 focus:border-indigo-500/50 focus:outline-none resize-none"
                                  />
                                  <div className="flex gap-1">
                                    <motion.button
                                      onClick={() => {
                                        if (!respondContent.trim()) return;
                                        setBankSection('log');
                                        bankAction(
                                          `RESPOND to msg ${msg.id?.slice(0, 8)}`,
                                          'POST',
                                          '/api/v1/quillon-bank/messages/admin/respond',
                                          { message_id: msg.id, wallet_address: msg.wallet_address, content: respondContent },
                                        );
                                        setRespondMsgId('');
                                        setRespondContent('');
                                      }}
                                      whileHover={{ scale: 1.05 }}
                                      whileTap={{ scale: 0.95 }}
                                      className="px-2 py-1 rounded text-[9px] text-emerald-300 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30"
                                    >
                                      Send
                                    </motion.button>
                                    <button
                                      onClick={() => setRespondMsgId('')}
                                      className="px-2 py-1 rounded text-[9px] text-amber-200/40 hover:text-amber-200/60"
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ─── Treasury Section ─── */}
                {bankSection === 'treasury' && (
                  <div className="space-y-3">
                    <div className="rounded-lg border border-slate-700/30 bg-slate-800/20 p-3">
                      <div className="flex items-center gap-1.5 mb-2">
                        <Wallet className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-[10px] font-semibold text-emerald-200/80 uppercase">Treasury Reserves</span>
                      </div>
                      {bankReserves ? (
                        <pre className="text-[10px] text-amber-200/60 font-mono bg-slate-900/50 rounded p-2 max-h-40 overflow-auto whitespace-pre-wrap">
                          {JSON.stringify(bankReserves, null, 2)}
                        </pre>
                      ) : (
                        <div className="text-center py-3 text-[10px] text-amber-200/30">Loading...</div>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { label: 'Fetch Reserves', cmd: 'GET treasury/reserves', path: '/api/v1/quillon-bank/treasury/reserves' },
                        { label: 'Fetch Profits', cmd: 'GET treasury/profits', path: '/api/v1/quillon-bank/treasury/profits' },
                        { label: 'Customer Analytics', cmd: 'GET analytics/customers', path: '/api/v1/quillon-bank/analytics/customers' },
                        { label: 'Dev Fee Stats', cmd: 'GET devfee/stats', path: '/api/v1/quillon-bank/devfee/stats' },
                      ].map(action => (
                        <motion.button
                          key={action.label}
                          onClick={() => { setBankSection('log'); bankAction(action.cmd, 'GET', action.path); }}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className="p-2 rounded-lg text-[10px] text-amber-200/60 hover:text-amber-200 bg-slate-700/20 hover:bg-slate-700/40 border border-slate-600/20 hover:border-amber-400/30 transition-all"
                        >
                          {action.label}
                        </motion.button>
                      ))}
                    </div>

                    {/* Execute liquidations */}
                    <motion.button
                      onClick={() => {
                        if (!confirm('Execute all pending liquidations?')) return;
                        setBankSection('log');
                        bankAction('EXECUTE LIQUIDATIONS', 'POST', '/api/v1/quillon-bank/risk/liquidations/execute', {});
                      }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className="w-full p-2 rounded-lg text-[10px] font-medium text-red-200/60 hover:text-red-200 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 hover:border-red-500/40 transition-all"
                    >
                      Execute Pending Liquidations
                    </motion.button>
                  </div>
                )}

                {/* ─── Command Log Section ─── */}
                {bankSection === 'log' && (
                  <div className="rounded-lg border border-slate-700/30 bg-slate-900/60 overflow-hidden">
                    <div className="flex items-center gap-1.5 px-3 py-2 bg-slate-800/50 border-b border-slate-700/30">
                      <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-[10px] font-semibold text-emerald-200/80 uppercase">Command Log</span>
                      <span className="ml-auto text-[9px] text-amber-200/30">{bankCmdLog.length} entries</span>
                      {bankCmdLog.length > 0 && (
                        <button
                          onClick={() => setBankCmdLog([])}
                          className="text-[9px] text-red-300/50 hover:text-red-300 transition-colors"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                    <div ref={bankLogRef} className="max-h-80 overflow-y-auto p-2 space-y-1.5 font-mono">
                      {bankCmdLog.length === 0 ? (
                        <div className="text-center py-6 text-[10px] text-amber-200/20">
                          Run a command to see results here
                        </div>
                      ) : (
                        bankCmdLog.map((entry, i) => (
                          <div key={i} className="space-y-0.5">
                            <div className="flex items-start gap-1.5">
                              <span className="text-emerald-400 text-[10px] select-none shrink-0">$</span>
                              <span className="text-[10px] text-amber-200/80 break-all">{entry.cmd}</span>
                              <span className="ml-auto text-[8px] text-amber-200/20 shrink-0">
                                {new Date(entry.ts).toLocaleTimeString()}
                              </span>
                            </div>
                            <pre className={`text-[9px] ml-4 p-1.5 rounded whitespace-pre-wrap break-all max-h-40 overflow-auto ${
                              entry.ok ? 'text-emerald-200/60 bg-emerald-500/5' : 'text-red-300/70 bg-red-500/5'
                            }`}>
                              {entry.result}
                            </pre>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* BRIDGE PAIRS TAB (v7.2.5)                                 */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'bridge' && (<>
                {/* Bridge Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-amber-400" />
                    <span className="text-xs font-bold text-amber-200/80 uppercase tracking-wider">
                      Cross-Chain Bridge Management
                    </span>
                  </div>
                  <motion.button
                    onClick={async () => {
                      try {
                        const resp = await fetch('/api/v1/bridge/status');
                        const data = await resp.json();
                        if (data.success) setBridgeData(data.data);
                      } catch {}
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] text-emerald-300/70 hover:text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 transition-all"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Refresh
                  </motion.button>
                </div>

                {/* Bridge Chain Cards */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { chain: 'Bitcoin', symbol: 'BTC', wrapped: 'wBTC', color: 'amber', icon: '₿', nodeIp: '5.79.79.158:8332', data: bridgeData?.bitcoin },
                    { chain: 'Zcash', symbol: 'ZEC', wrapped: 'wZEC', color: 'blue', icon: '🛡', nodeIp: '5.79.79.158:8232', data: bridgeData?.zcash },
                    { chain: 'Iron Fish', symbol: 'IRON', wrapped: 'wIRON', color: 'cyan', icon: '🐟', nodeIp: '5.79.79.158:8021', data: bridgeData?.ironfish },
                    { chain: 'Ethereum', symbol: 'ETH', wrapped: 'wETH', color: 'indigo', icon: '⟠', nodeIp: '5.79.79.158:8545', data: bridgeData?.ethereum },
                  ].map(bridge => (
                    <div key={bridge.chain} className={`rounded-xl border border-${bridge.color}-500/30 bg-${bridge.color}-500/5 p-3`}>
                      <div className="flex items-center gap-2 mb-3">
                        <div className={`w-8 h-8 rounded-lg bg-${bridge.color}-500/20 flex items-center justify-center text-lg`}>
                          {bridge.icon}
                        </div>
                        <div>
                          <div className="text-xs font-bold text-white">{bridge.chain}</div>
                          <div className="text-[10px] text-gray-400">{bridge.wrapped} / QUG</div>
                        </div>
                        <div className="ml-auto">
                          <div className={`w-2 h-2 rounded-full ${bridge.data?.node_connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
                        </div>
                      </div>

                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">Node</span>
                          <span className={bridge.data?.node_connected ? 'text-emerald-300' : 'text-red-400'}>
                            {bridge.data?.node_connected ? 'Connected' : 'Offline'}
                          </span>
                        </div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">Synced</span>
                          <span className={bridge.data?.node_synced ? 'text-emerald-300' : 'text-yellow-300'}>
                            {bridge.data?.node_synced ? 'Yes' : 'Syncing...'}
                          </span>
                        </div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">Total Locked</span>
                          <span className="text-white font-mono">
                            {(bridge.data?.total_locked || 0).toFixed(bridge.symbol === 'BTC' ? 8 : 4)} {bridge.symbol}
                          </span>
                        </div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">Total Minted</span>
                          <span className="text-amber-300 font-mono">
                            {(bridge.data?.total_minted || 0).toFixed(bridge.symbol === 'BTC' ? 8 : 4)} {bridge.wrapped}
                          </span>
                        </div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">Active Bridges</span>
                          <span className="text-white">{bridge.data?.active_bridges || 0}</span>
                        </div>
                        <div className="flex justify-between text-[10px]">
                          <span className="text-gray-400">RPC Endpoint</span>
                          <span className="text-gray-500 font-mono text-[8px]">{bridge.nodeIp}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Bridge Pool Management */}
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                  <div className="flex items-center gap-2 mb-3">
                    <Database className="w-4 h-4 text-amber-400" />
                    <span className="text-xs font-bold text-amber-200">Bridge Liquidity Pools</span>
                  </div>
                  <div className="space-y-2">
                    {[
                      { pair: 'wBTC / QUG', poolId: 'pool-qug-wbtc-bridge', color: 'amber' },
                      { pair: 'wZEC / QUG', poolId: 'pool-qug-wzec-bridge', color: 'blue' },
                      { pair: 'wIRON / QUG', poolId: 'pool-qug-wiron-bridge', color: 'cyan' },
                      { pair: 'wETH / QUG', poolId: 'pool-qug-weth-bridge', color: 'indigo' },
                    ].map(pool => (
                      <div key={pool.poolId} className="flex items-center justify-between bg-slate-800/40 rounded-lg p-2.5">
                        <div className="flex items-center gap-2">
                          <div className={`w-6 h-6 rounded-full bg-${pool.color}-500/30 flex items-center justify-center text-[10px]`}>
                            {pool.pair.split(' ')[0].charAt(0)}
                          </div>
                          <div>
                            <div className="text-xs font-bold text-white">{pool.pair}</div>
                            <div className="text-[9px] text-gray-500 font-mono">{pool.poolId}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-[9px] text-emerald-300 font-medium">ACTIVE</span>
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-[9px] text-amber-300 font-medium">MASTER</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-3 pt-2 border-t border-amber-500/10">
                    <p className="text-[9px] text-amber-200/40 italic">
                      Bridge pools are system-owned by the master wallet. Liquidity is automatically managed
                      through the mint/burn bridge mechanism. Pools were bootstrapped at node startup.
                    </p>
                  </div>
                </div>

                {/* Multi-Sig Committee Status (v7.3.2) */}
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-bold text-emerald-200">Multi-Sig Bridge Committee</span>
                    <span className="ml-auto px-1.5 py-0.5 rounded bg-emerald-500/20 text-[9px] text-emerald-300 font-medium">
                      7-of-11 THRESHOLD
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="bg-slate-800/40 rounded-lg p-2 text-center">
                      <div className="text-lg font-bold text-emerald-300">{bridgeData?.committee?.size || 11}</div>
                      <div className="text-[9px] text-gray-400">Committee Size</div>
                    </div>
                    <div className="bg-slate-800/40 rounded-lg p-2 text-center">
                      <div className="text-lg font-bold text-amber-300">{bridgeData?.committee?.epoch || '--'}</div>
                      <div className="text-[9px] text-gray-400">Current Epoch</div>
                    </div>
                    <div className="bg-slate-800/40 rounded-lg p-2 text-center">
                      <div className="text-lg font-bold text-purple-300">{bridgeData?.committee?.attestations_total || 0}</div>
                      <div className="text-[9px] text-gray-400">Total Attestations</div>
                    </div>
                  </div>

                  {/* Committee Members */}
                  <div className="space-y-1 mb-3">
                    <div className="text-[10px] text-gray-400 font-medium mb-1">Active Committee Members</div>
                    {(bridgeData?.committee?.members || []).length > 0 ? (
                      (bridgeData?.committee?.members || []).slice(0, 5).map((member: any, i: number) => (
                        <div key={i} className="flex items-center justify-between bg-slate-800/30 rounded px-2 py-1">
                          <div className="flex items-center gap-1.5">
                            <div className={`w-1.5 h-1.5 rounded-full ${member.online ? 'bg-emerald-400' : 'bg-red-500'}`} />
                            <span className="text-[9px] text-gray-300 font-mono">{(member.peer_id || '').slice(0, 16)}...</span>
                          </div>
                          <span className="text-[9px] text-emerald-300">{member.attestations || 0} att.</span>
                        </div>
                      ))
                    ) : (
                      <div className="text-[9px] text-gray-500 italic">Committee initializing... (need 7+ peers)</div>
                    )}
                    {(bridgeData?.committee?.members || []).length > 5 && (
                      <div className="text-[9px] text-gray-500 text-center">
                        +{(bridgeData?.committee?.members || []).length - 5} more members
                      </div>
                    )}
                  </div>

                  {/* Attestation Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pending Claims</span>
                      <span className="text-amber-300">{bridgeData?.committee?.pending_claims || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Approved Claims</span>
                      <span className="text-emerald-300">{bridgeData?.committee?.approved_claims || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Rejected Claims</span>
                      <span className="text-red-400">{bridgeData?.committee?.rejected_claims || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Avg Response</span>
                      <span className="text-gray-300">{bridgeData?.committee?.avg_response_ms || '--'}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Epoch Rotation</span>
                      <span className="text-gray-300">Every 100 blocks</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Mode</span>
                      <span className={`${(bridgeData?.committee?.members || []).length >= 7 ? 'text-emerald-300' : 'text-yellow-300'}`}>
                        {(bridgeData?.committee?.members || []).length >= 7 ? 'Multi-Sig' : 'Single-Node Fallback'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bridge Architecture Diagram */}
                <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Layers className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-bold text-purple-200">Decentralized Bridge Architecture (v2.0)</span>
                  </div>
                  <pre className="text-[9px] text-slate-400 font-mono leading-relaxed whitespace-pre">
{`  Bitcoin     Zcash      Iron Fish    Ethereum
  (Knots)    (Zebra)      (Node)      (Geth)
     |          |            |           |
  [Lock]     [Lock]      [Lock]      [Lock ETH]
     |          |            |           |
     └──────────┴────────────┴───────────┘
                      |
              ┌───────┴───────┐
              │  7-of-11      │
              │  Committee    │ ← Gossipsub Attestations
              │  Validation   │   (CBOR/Ed25519 signed)
              └───────┬───────┘
                      |
         ┌────────────┼────────────┐
         │    QNK Bridge Engine    │
         │   Mint/Burn + AMM DEX  │
         └─┬─────┬─────┬──────┬───┘
           │     │     │      │
       wBTC  wZEC  wIRON  wETH  / QUG
        Pool  Pool  Pool   Pool`}
                  </pre>
                </div>

                {/* Bridge Server Info */}
                <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Server className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-bold text-indigo-200">Server Delta (Bridge Nodes)</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                    <div className="flex justify-between">
                      <span className="text-gray-400">IP Address</span>
                      <span className="text-white font-mono">5.79.79.158</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Role</span>
                      <span className="text-amber-300">Bridge Node Host</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bitcoin Knots</span>
                      <span className="text-amber-300 font-mono">:8332</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Zcash Zebra</span>
                      <span className="text-blue-300 font-mono">:8232</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Iron Fish</span>
                      <span className="text-cyan-300 font-mono">:8021</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Storage</span>
                      <span className="text-gray-300">/home/data/ (7.2TB)</span>
                    </div>
                  </div>
                </div>
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* BOUNTY ADMIN TAB                                          */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'bounty' && (<>
                {bountyLoading && !bountyStats ? (
                  <div className="flex items-center justify-center py-16">
                    <RefreshCw className="w-6 h-6 text-emerald-400 animate-spin" />
                  </div>
                ) : (
                  <>
                    {/* Stats Summary */}
                    {bountyStats && (
                      <div className="grid grid-cols-5 gap-2">
                        {[
                          { label: 'Users', value: bountyStats.total_users, color: 'text-emerald-400' },
                          { label: 'Bug Reports', value: bountyStats.total_bug_reports, color: 'text-blue-400' },
                          { label: 'Pending Bugs', value: bountyStats.pending_bug_reports, color: 'text-amber-400' },
                          { label: 'Social Posts', value: bountyStats.total_social_activities, color: 'text-purple-400' },
                          { label: 'Pending Social', value: bountyStats.pending_social_activities, color: 'text-pink-400' },
                        ].map(s => (
                          <div key={s.label} className="rounded-lg border border-slate-700/50 bg-slate-800/30 p-3 text-center">
                            <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
                            <div className="text-[10px] text-amber-200/40 uppercase tracking-wider">{s.label}</div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Sub-tabs: Bugs / Social */}
                    <div className="flex items-center gap-2">
                      {(['bugs', 'social'] as const).map(t => (
                        <button
                          key={t}
                          onClick={() => setBountyTab(t)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                            bountyTab === t
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : 'text-amber-200/50 hover:text-amber-200/80 hover:bg-slate-800/30'
                          }`}
                        >
                          {t === 'bugs' ? 'Bug Reports' : 'Social Activities'}
                        </button>
                      ))}
                      <button
                        onClick={fetchBountyData}
                        className="ml-auto p-1.5 rounded-lg text-amber-200/40 hover:text-emerald-300 hover:bg-emerald-500/10 transition-all"
                        title="Refresh"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${bountyLoading ? 'animate-spin' : ''}`} />
                      </button>
                    </div>

                    {/* Bug Reports Table */}
                    {bountyTab === 'bugs' && (
                      <div className="space-y-2">
                        {bountyBugs.length === 0 ? (
                          <div className="text-center py-8 text-amber-200/30 text-xs">No bug reports submitted yet</div>
                        ) : bountyBugs.map((bug: any, i: number) => (
                          <div key={i} className="rounded-lg border border-slate-700/40 bg-slate-800/20 p-3">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                                    bug.severity === 'Critical' ? 'bg-red-500/20 text-red-400' :
                                    bug.severity === 'High' ? 'bg-orange-500/20 text-orange-400' :
                                    bug.severity === 'Medium' ? 'bg-amber-500/20 text-amber-400' :
                                    'bg-slate-500/20 text-slate-400'
                                  }`}>{bug.severity}</span>
                                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                    bug.status === 'Submitted' ? 'bg-blue-500/20 text-blue-300' :
                                    bug.status === 'Verified' ? 'bg-emerald-500/20 text-emerald-300' :
                                    bug.status === 'Fixed' ? 'bg-green-500/20 text-green-300' :
                                    bug.status === 'Duplicate' || bug.status === 'Invalid' ? 'bg-red-500/20 text-red-300' :
                                    'bg-amber-500/20 text-amber-300'
                                  }`}>{bug.status}</span>
                                  <span className="text-[10px] text-amber-200/30">{bug.points} pts</span>
                                </div>
                                <div className="text-xs text-white/80 truncate">{bug.description || 'No description'}</div>
                                <div className="flex items-center gap-3 mt-1">
                                  {bug.issue_url && (
                                    <a href={bug.issue_url} target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-400 hover:underline truncate max-w-[200px]">
                                      {bug.issue_url}
                                    </a>
                                  )}
                                  <span className="text-[10px] text-amber-200/20">
                                    User: {bug.user_id?.slice(0, 8)}...
                                  </span>
                                  <span className="text-[10px] text-amber-200/20">
                                    {new Date(bug.timestamp).toLocaleDateString()}
                                  </span>
                                </div>
                              </div>
                              {/* Action buttons */}
                              <div className="flex items-center gap-1 flex-shrink-0">
                                {bug.status === 'Submitted' && (
                                  <>
                                    <button
                                      onClick={() => updateBugStatus(bug.user_id, bug.timestamp, 'Verified')}
                                      className="px-2 py-1 rounded text-[10px] font-medium bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 transition-all"
                                      title="Approve"
                                    >
                                      <CheckCircle className="w-3 h-3" />
                                    </button>
                                    <button
                                      onClick={() => updateBugStatus(bug.user_id, bug.timestamp, 'Invalid')}
                                      className="px-2 py-1 rounded text-[10px] font-medium bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all"
                                      title="Reject"
                                    >
                                      <XCircle className="w-3 h-3" />
                                    </button>
                                    <button
                                      onClick={() => updateBugStatus(bug.user_id, bug.timestamp, 'Duplicate')}
                                      className="px-2 py-1 rounded text-[10px] font-medium bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 transition-all"
                                      title="Duplicate"
                                    >
                                      <Copy className="w-3 h-3" />
                                    </button>
                                  </>
                                )}
                                {bug.status === 'Verified' && (
                                  <button
                                    onClick={() => updateBugStatus(bug.user_id, bug.timestamp, 'Fixed')}
                                    className="px-2 py-1 rounded text-[10px] font-medium bg-green-500/20 text-green-300 hover:bg-green-500/30 transition-all"
                                    title="Mark Fixed"
                                  >
                                    Fixed
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Social Activities Table */}
                    {bountyTab === 'social' && (
                      <div className="space-y-2">
                        {bountySocials.length === 0 ? (
                          <div className="text-center py-8 text-amber-200/30 text-xs">No social activities submitted yet</div>
                        ) : bountySocials.map((social: any, i: number) => (
                          <div key={i} className="rounded-lg border border-slate-700/40 bg-slate-800/20 p-3">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                    social.platform === 'Twitter' ? 'bg-sky-500/20 text-sky-400' :
                                    social.platform === 'GitHub' ? 'bg-slate-500/20 text-slate-300' :
                                    social.platform === 'Discord' ? 'bg-indigo-500/20 text-indigo-400' :
                                    social.platform === 'YouTube' ? 'bg-red-500/20 text-red-400' :
                                    'bg-purple-500/20 text-purple-400'
                                  }`}>{social.platform}</span>
                                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-600/30 text-slate-300">{social.activity_type}</span>
                                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                    social.verified ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                                  }`}>{social.verified ? 'Verified' : 'Pending'}</span>
                                  <span className="text-[10px] text-amber-200/30">{social.engagement_score} pts</span>
                                </div>
                                <div className="flex items-center gap-3 mt-1">
                                  {social.content_url && (
                                    <a href={social.content_url} target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-400 hover:underline truncate max-w-[300px]">
                                      {social.content_url}
                                    </a>
                                  )}
                                  <span className="text-[10px] text-amber-200/20">
                                    User: {social.user_id?.slice(0, 8)}...
                                  </span>
                                  <span className="text-[10px] text-amber-200/20">
                                    {new Date(social.timestamp).toLocaleDateString()}
                                  </span>
                                </div>
                              </div>
                              {/* Action buttons */}
                              <div className="flex items-center gap-1 flex-shrink-0">
                                {!social.verified && (
                                  <>
                                    <button
                                      onClick={() => {
                                        const platformMap: Record<string, number> = { Twitter: 0, GitHub: 1, Discord: 2, Medium: 3, YouTube: 4 };
                                        updateSocialStatus(social.user_id, platformMap[social.platform] ?? 0, social.timestamp, true);
                                      }}
                                      className="px-2 py-1 rounded text-[10px] font-medium bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 transition-all"
                                      title="Approve"
                                    >
                                      <CheckCircle className="w-3 h-3" />
                                    </button>
                                    <button
                                      onClick={() => {
                                        const platformMap: Record<string, number> = { Twitter: 0, GitHub: 1, Discord: 2, Medium: 3, YouTube: 4 };
                                        updateSocialStatus(social.user_id, platformMap[social.platform] ?? 0, social.timestamp, false);
                                      }}
                                      className="px-2 py-1 rounded text-[10px] font-medium bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all"
                                      title="Reject"
                                    >
                                      <XCircle className="w-3 h-3" />
                                    </button>
                                  </>
                                )}
                                {social.verified && (
                                  <span className="text-[10px] text-emerald-400 flex items-center gap-1">
                                    <BadgeCheck className="w-3 h-3" /> Approved
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* DEX ANALYTICS TAB (v8.5.3)                                */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'dex' && (<>
                {/* Sub-tabs */}
                <div className="flex items-center gap-2 mb-4">
                  {(['analytics', 'pools', 'fees'] as const).map(sec => (
                    <button key={sec} onClick={() => setDexSection(sec)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        dexSection === sec ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }`}>
                      {sec === 'analytics' ? 'Overview' : sec === 'pools' ? 'Pools' : 'Fee Revenue'}
                    </button>
                  ))}
                  <button onClick={fetchDexData} disabled={dexLoading}
                    className="ml-auto p-1.5 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all">
                    <RefreshCw className={`w-3.5 h-3.5 ${dexLoading ? 'animate-spin' : ''}`} />
                  </button>
                </div>

                {dexLoading && dexPools.length === 0 ? (
                  <div className="flex items-center justify-center py-16">
                    <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
                  </div>
                ) : (<>

                {/* Analytics Overview */}
                {dexSection === 'analytics' && (
                  <div className="space-y-4">
                    {/* Top metrics */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Layers className="w-3.5 h-3.5 text-cyan-400" />
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider">Active Pools</span>
                        </div>
                        <div className="text-xl font-bold text-white">{dexPools.length}</div>
                      </div>
                      <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider">Total Liquidity</span>
                        </div>
                        <div className="text-xl font-bold text-white">
                          ${dexPools.reduce((sum, p) => {
                            const r0 = parseFloat(p.reserve0 || p.total_liquidity || '0');
                            const r1 = parseFloat(p.reserve1 || '0');
                            return sum + r0 + r1;
                          }, 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </div>
                      </div>
                      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Activity className="w-3.5 h-3.5 text-amber-400" />
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider">Protocol Fee</span>
                        </div>
                        <div className="text-xl font-bold text-white">
                          {dexFeeStats?.dex_protocol_fee_percent || '0.15%'}
                        </div>
                        <div className="text-[10px] text-gray-500">of 0.30% total</div>
                      </div>
                      <div className="rounded-xl border border-purple-500/20 bg-purple-500/5 p-3">
                        <div className="flex items-center gap-2 mb-1">
                          <Wallet className="w-3.5 h-3.5 text-purple-400" />
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider">Founder Balance</span>
                        </div>
                        <div className="text-xl font-bold text-white">
                          {(dexFeeStats?.founder_wallet_balance_qug || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG
                        </div>
                      </div>
                    </div>

                    {/* Fee structure explanation */}
                    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                      <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                        <DollarSign className="w-4 h-4 text-emerald-400" />
                        Fee Structure (per swap)
                      </h4>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="text-center p-3 rounded-lg bg-black/30 border border-cyan-500/10">
                          <div className="text-lg font-bold text-cyan-300">0.30%</div>
                          <div className="text-[10px] text-gray-400">Total Fee</div>
                        </div>
                        <div className="text-center p-3 rounded-lg bg-black/30 border border-emerald-500/10">
                          <div className="text-lg font-bold text-emerald-300">0.15%</div>
                          <div className="text-[10px] text-gray-400">→ LP Providers</div>
                        </div>
                        <div className="text-center p-3 rounded-lg bg-black/30 border border-amber-500/10">
                          <div className="text-lg font-bold text-amber-300">0.15%</div>
                          <div className="text-[10px] text-gray-400">→ Protocol Treasury</div>
                        </div>
                      </div>
                    </div>

                    {/* Operator fee settings */}
                    {dexFeeStats && (
                      <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                        <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                          <Settings className="w-4 h-4 text-gray-400" />
                          Node Operator Fee Settings
                        </h4>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-gray-400">Operator Fee:</span>
                            <span className="text-white ml-2 font-mono">{dexFeeStats.node_operator_fee_percent || '0%'}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Protocol Fee:</span>
                            <span className="text-white ml-2 font-mono">{dexFeeStats.dex_protocol_fee_percent || '0.15%'}</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Admin Wallet:</span>
                            <span className="text-cyan-300 ml-2 font-mono text-[10px]">{(dexFeeStats.admin_wallet || '').slice(0, 12)}...</span>
                          </div>
                          <div>
                            <span className="text-gray-400">Admin Balance:</span>
                            <span className="text-white ml-2 font-mono">{(dexFeeStats.admin_wallet_balance_qug || 0).toFixed(4)} QUG</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Pools Tab */}
                {dexSection === 'pools' && (
                  <div className="space-y-3">
                    {dexPools.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">No pools found</div>
                    ) : dexPools.map((pool: any, i: number) => (
                      <div key={i} className="rounded-xl border border-white/10 bg-white/5 p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center text-white text-xs font-bold">
                              {(pool.token0 || 'T0').slice(0, 2)}
                            </div>
                            <div>
                              <div className="text-sm font-bold text-white">
                                {pool.token0 || 'Token0'} / {pool.token1 || 'Token1'}
                              </div>
                              <div className="text-[10px] text-gray-500 font-mono">{pool.address || pool.pool_id || `pool-${i}`}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-400">Fee</div>
                            <div className="text-sm font-bold text-amber-300">{((pool.fee || 30) / 100).toFixed(2)}%</div>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="p-2 rounded-lg bg-black/30">
                            <div className="text-[10px] text-gray-500">{pool.token0 || 'Reserve 0'}</div>
                            <div className="text-sm font-mono text-white">
                              {parseFloat(pool.reserve0 || '0').toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </div>
                          </div>
                          <div className="p-2 rounded-lg bg-black/30">
                            <div className="text-[10px] text-gray-500">{pool.token1 || 'Reserve 1'}</div>
                            <div className="text-sm font-mono text-white">
                              {parseFloat(pool.reserve1 || '0').toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </div>
                          </div>
                        </div>
                        {pool.reserve0 && pool.reserve1 && parseFloat(pool.reserve0) > 0 && (
                          <div className="mt-2 text-xs text-gray-400">
                            Price: 1 {pool.token0} = {(parseFloat(pool.reserve1) / parseFloat(pool.reserve0)).toFixed(4)} {pool.token1}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Fee Revenue Tab */}
                {dexSection === 'fees' && (
                  <div className="space-y-4">
                    <div className="rounded-xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-cyan-500/5 p-4">
                      <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                        <Banknote className="w-4 h-4 text-emerald-400" />
                        Protocol Fee Revenue
                      </h4>
                      <div className="text-[10px] text-gray-400 mb-4">
                        0.15% of every swap goes to the protocol treasury (master wallet). 0.15% goes to LP providers.
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 rounded-lg bg-black/30 border border-emerald-500/10">
                          <div className="text-[10px] text-gray-500 mb-1">Founder Wallet Balance</div>
                          <div className="text-lg font-bold text-emerald-300">
                            {(dexFeeStats?.founder_wallet_balance_qug || 0).toLocaleString(undefined, { maximumFractionDigits: 4 })} QUG
                          </div>
                        </div>
                        <div className="p-3 rounded-lg bg-black/30 border border-cyan-500/10">
                          <div className="text-[10px] text-gray-500 mb-1">Admin Wallet Balance</div>
                          <div className="text-lg font-bold text-cyan-300">
                            {(dexFeeStats?.admin_wallet_balance_qug || 0).toLocaleString(undefined, { maximumFractionDigits: 4 })} QUG
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Fee collection per pool */}
                    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                      <h4 className="text-sm font-bold text-white mb-3">Fee Collection by Pool</h4>
                      {dexPools.length === 0 ? (
                        <div className="text-center py-4 text-gray-500 text-sm">No pool data</div>
                      ) : (
                        <div className="space-y-2">
                          {dexPools.map((pool: any, i: number) => {
                            const vol = parseFloat(pool.volume_24h || '0');
                            const feeRate = (pool.fee || 30) / 10000;
                            const protocolFee = vol * feeRate * 0.5; // 50% of fees to protocol
                            return (
                              <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-black/20">
                                <span className="text-sm text-white font-medium">
                                  {pool.token0}/{pool.token1}
                                </span>
                                <div className="text-right">
                                  <div className="text-xs text-gray-400">24h Vol: ${vol.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                                  <div className="text-xs text-emerald-400">Protocol: ${protocolFee.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    {/* Sound money info */}
                    <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                      <h4 className="text-sm font-bold text-amber-300 mb-2 flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        Sound Money Guarantee
                      </h4>
                      <ul className="text-xs text-gray-300 space-y-1">
                        <li>21,000,000 QUG maximum supply — enforced by emission controller</li>
                        <li>4-year halving schedule (64 eras, 256 years to full emission)</li>
                        <li>Every block signed with Dilithium5 post-quantum signatures</li>
                        <li>No pre-mine, no ICO, no allocation — everyone started at genesis</li>
                        <li>Pool reserves verified by constant-product AMM (x * y = k)</li>
                      </ul>
                    </div>
                  </div>
                )}

                </>)}
              </>)}

              {/* ═══════════════════════════════════════════════════════════ */}
              {/* NODE SETTINGS TAB (v7.3.0)                                */}
              {/* ═══════════════════════════════════════════════════════════ */}
              {activeTab === 'settings' && (<>
                {settingsLoading && !adminSettings ? (
                  <div className="flex items-center justify-center py-16">
                    <RefreshCw className="w-6 h-6 text-emerald-400 animate-spin" />
                  </div>
                ) : adminSettings ? (<>
                  {/* ── Node Identity Card ── */}
                  <div className="rounded-xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-indigo-500/5 p-4 relative overflow-hidden">
                    {/* Decorative circuit pattern */}
                    <div className="absolute top-0 right-0 w-24 h-24 opacity-5">
                      <svg viewBox="0 0 100 100" fill="none" stroke="currentColor" className="text-emerald-400 w-full h-full">
                        <circle cx="50" cy="50" r="20" strokeWidth="1"/>
                        <circle cx="50" cy="50" r="35" strokeWidth="0.5"/>
                        <line x1="50" y1="0" x2="50" y2="30" strokeWidth="0.5"/>
                        <line x1="50" y1="70" x2="50" y2="100" strokeWidth="0.5"/>
                        <line x1="0" y1="50" x2="30" y2="50" strokeWidth="0.5"/>
                        <line x1="70" y1="50" x2="100" y2="50" strokeWidth="0.5"/>
                        <line x1="15" y1="15" x2="35" y2="35" strokeWidth="0.5"/>
                        <line x1="65" y1="65" x2="85" y2="85" strokeWidth="0.5"/>
                      </svg>
                    </div>
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/30 to-indigo-500/30 border border-emerald-500/40 flex items-center justify-center">
                        <Cpu className="w-5 h-5 text-emerald-300" />
                      </div>
                      <div>
                        <div className="text-xs text-amber-200/50 uppercase tracking-wider font-semibold">Node Operator</div>
                        <div className="text-sm text-white font-mono">{adminSettings.admin_wallet}</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2 mt-3">
                      <div className="text-center p-2 rounded-lg bg-black/20">
                        <div className="text-base font-bold text-emerald-300">v{adminSettings.version}</div>
                        <div className="text-[10px] text-amber-200/40 uppercase">Version</div>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-black/20">
                        <div className="text-base font-bold text-white">{(() => {
                          const s = adminSettings.uptime_secs;
                          const d = Math.floor(s / 86400);
                          const h = Math.floor((s % 86400) / 3600);
                          return d > 0 ? `${d}d ${h}h` : `${h}h ${Math.floor((s % 3600) / 60)}m`;
                        })()}</div>
                        <div className="text-[10px] text-amber-200/40 uppercase">Uptime</div>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-black/20">
                        <div className="text-base font-bold text-indigo-300">{adminSettings.peers}</div>
                        <div className="text-[10px] text-amber-200/40 uppercase">Peers</div>
                      </div>
                    </div>
                  </div>

                  {/* ── Sync Progress ── */}
                  <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4 text-indigo-400" />
                        <span className="text-xs font-semibold text-indigo-200">Chain Sync</span>
                      </div>
                      <span className="text-xs text-amber-200/40 font-mono">{adminSettings.network_id}</span>
                    </div>
                    <div className="flex items-end gap-3">
                      <div className="flex-1">
                        <div className="flex justify-between text-[10px] text-amber-200/50 mb-1">
                          <span>{adminSettings.height.toLocaleString()} blocks</span>
                          <span>{adminSettings.network_height.toLocaleString()} network</span>
                        </div>
                        <div className="w-full h-2.5 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-700"
                            style={{
                              width: `${adminSettings.network_height > 0 ? Math.min(100, (adminSettings.height / adminSettings.network_height) * 100) : 100}%`,
                              background: adminSettings.height >= adminSettings.network_height * 0.995
                                ? 'linear-gradient(90deg, #22c55e, #4ade80)'
                                : 'linear-gradient(90deg, #6366f1, #818cf8)',
                            }}
                          />
                        </div>
                      </div>
                      <span className={`text-sm font-bold ${
                        adminSettings.height >= adminSettings.network_height * 0.995 ? 'text-green-400' : 'text-indigo-300'
                      }`}>
                        {adminSettings.network_height > 0
                          ? `${((adminSettings.height / adminSettings.network_height) * 100).toFixed(1)}%`
                          : '100%'
                        }
                      </span>
                    </div>
                  </div>

                  {/* ── Node Health Indicators ── */}
                  {nodeInfo && (
                    <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs font-semibold text-emerald-200">Health Monitor</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          {
                            label: 'Mining',
                            ok: nodeInfo.mining_healthy,
                            icon: Zap,
                            detail: nodeInfo.mining_healthy ? 'Solutions arriving' : 'No recent solutions',
                          },
                          {
                            label: 'P2P Network',
                            ok: nodeInfo.peers > 0,
                            icon: Wifi,
                            detail: `${nodeInfo.peers} peer${nodeInfo.peers !== 1 ? 's' : ''} connected`,
                          },
                          {
                            label: 'Block Sync',
                            ok: nodeInfo.height >= nodeInfo.network_height * 0.99,
                            icon: Layers,
                            detail: nodeInfo.height >= nodeInfo.network_height ? 'Fully synced' : `${(nodeInfo.network_height - nodeInfo.height).toLocaleString()} behind`,
                          },
                          {
                            label: 'Uptime',
                            ok: nodeInfo.uptime_secs > 300,
                            icon: Clock,
                            detail: nodeInfo.uptime_secs > 86400 ? `${Math.floor(nodeInfo.uptime_secs / 86400)}d stable` : 'Recently started',
                          },
                        ].map(item => (
                          <div
                            key={item.label}
                            className={`flex items-center gap-2.5 p-2.5 rounded-lg border ${
                              item.ok
                                ? 'border-emerald-500/20 bg-emerald-500/5'
                                : 'border-amber-500/20 bg-amber-500/5'
                            }`}
                          >
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                              item.ok ? 'bg-emerald-500/20' : 'bg-amber-500/20'
                            }`}>
                              <item.icon className={`w-4 h-4 ${item.ok ? 'text-emerald-400' : 'text-amber-400'}`} />
                            </div>
                            <div className="min-w-0">
                              <div className="text-[10px] text-amber-200/50 uppercase tracking-wider">{item.label}</div>
                              <div className={`text-xs font-medium ${item.ok ? 'text-emerald-300' : 'text-amber-300'}`}>
                                {item.detail}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ── OAuth2 Consent Manager ── */}
                  <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Lock className="w-4 h-4 text-violet-400" />
                        <span className="text-xs font-semibold text-violet-200">OAuth2 Integrations</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1.5">
                          <div className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                          <span className="text-[10px] text-amber-200/40">{adminSettings.oauth2_clients} client{adminSettings.oauth2_clients !== 1 ? 's' : ''}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="text-[10px] text-amber-200/40">{adminSettings.oauth2_active_tokens} active token{adminSettings.oauth2_active_tokens !== 1 ? 's' : ''}</span>
                        </div>
                      </div>
                    </div>

                    {oauthConsents.length === 0 ? (
                      <div className="text-center py-6 border border-dashed border-slate-700/50 rounded-lg">
                        <Fingerprint className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                        <div className="text-xs text-amber-200/30">No third-party apps authorized</div>
                        <div className="text-[10px] text-amber-200/20 mt-1">OAuth2 consents will appear here</div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {oauthConsents.map((c: any, i: number) => (
                          <div
                            key={i}
                            className="flex items-center justify-between p-3 rounded-lg bg-black/20 border border-slate-700/30 group hover:border-violet-500/30 transition-colors"
                          >
                            <div className="flex items-center gap-3 min-w-0">
                              <div className="w-8 h-8 rounded-lg bg-violet-500/15 border border-violet-500/20 flex items-center justify-center flex-shrink-0">
                                <Key className="w-3.5 h-3.5 text-violet-400" />
                              </div>
                              <div className="min-w-0">
                                <div className="text-xs font-medium text-white truncate">{c.client_id}</div>
                                <div className="text-[10px] text-amber-200/30 mt-0.5">
                                  {c.scopes?.join(' ') || 'default'} &middot; {new Date(c.granted_at).toLocaleDateString()}
                                </div>
                              </div>
                            </div>
                            <button
                              onClick={async () => {
                                await fetch('/api/v1/admin/oauth2/revoke-consent', {
                                  method: 'POST',
                                  headers: { 'X-Wallet-Auth': walletAddress, 'Authorization': `Bearer ${walletAddress}`, 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ client_id: c.client_id }),
                                });
                                fetchSettingsData();
                              }}
                              className="p-1.5 rounded-lg text-red-400/50 hover:text-red-400 hover:bg-red-500/15 opacity-0 group-hover:opacity-100 transition-all"
                              title="Revoke consent & tokens"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* ── CLI Quick Reference ── */}
                  <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Terminal className="w-4 h-4 text-amber-400" />
                      <span className="text-xs font-semibold text-amber-200/80">CLI Reference</span>
                    </div>
                    <div className="space-y-1.5">
                      {[
                        { cmd: '--admin-wallet <hex>', desc: 'Set admin wallet for this node' },
                        { cmd: 'Q_ADMIN_WALLET=<hex>', desc: 'Environment variable alternative' },
                        { cmd: '--validator-key <path>', desc: 'Enable PQC block signing' },
                      ].map(item => (
                        <div key={item.cmd} className="flex items-start gap-2">
                          <Hash className="w-3 h-3 text-amber-400/40 mt-0.5 flex-shrink-0" />
                          <div>
                            <code className="text-[11px] text-emerald-300 font-mono">{item.cmd}</code>
                            <div className="text-[10px] text-amber-200/30">{item.desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>) : (
                  <div className="text-center py-12 text-amber-200/30 text-xs">
                    Could not load settings. Make sure your wallet matches --admin-wallet.
                  </div>
                )}
              </>)}

              {/* v9.1.4: Mining Mode Control Tab */}
              {/* v9.3.1: K-Parameter Health Gauge Tab */}
              {activeTab === 'kparam' && (<>
                <div className="space-y-4">
                  {/* Header with refresh */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm font-semibold text-cyan-200">Network Health Gauge</span>
                      <span className="text-[10px] text-amber-200/30 font-mono">K = 2π √(ΔH · Δs · ℏ) / τ</span>
                    </div>
                    <button onClick={fetchKParamData} disabled={kParamLoading} className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                      <RefreshCw className={`w-3.5 h-3.5 text-emerald-300/50 ${kParamLoading ? 'animate-spin' : ''}`} />
                    </button>
                  </div>

                  {!kParamData ? (
                    <div className="text-center py-8">
                      {kParamLoading ? (
                        <div className="flex items-center justify-center gap-2 text-amber-200/40 text-xs">
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Loading K-parameter data...
                        </div>
                      ) : kParamError ? (
                        <div className="space-y-3">
                          <div className="flex items-center justify-center gap-2 text-red-400/70 text-xs">
                            <AlertTriangle className="w-4 h-4" />
                            {kParamError}
                          </div>
                          <button onClick={fetchKParamData} className="px-4 py-1.5 rounded-lg text-xs bg-slate-700/50 text-emerald-300/70 hover:bg-slate-700/80 border border-slate-600/30 transition-colors">
                            Retry
                          </button>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <div className="text-amber-200/30 text-xs">No data yet — gauge starts reporting after 60 seconds</div>
                          <button onClick={fetchKParamData} className="px-4 py-1.5 rounded-lg text-xs bg-slate-700/50 text-emerald-300/70 hover:bg-slate-700/80 border border-slate-600/30 transition-colors">
                            Fetch Now
                          </button>
                        </div>
                      )}
                    </div>
                  ) : (<>
                    {/* v9.3.2: Orbital K-Value Visualization with educational info panel */}
                    <div className="relative">
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className="rounded-xl overflow-hidden cursor-pointer group"
                        style={{
                          background: 'rgba(5, 10, 25, 0.85)',
                          border: `1px solid ${kParamData.phase === 'critical' ? 'rgba(239, 68, 68, 0.4)' : kParamData.phase === 'approaching' ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.25)'}`,
                          boxShadow: kParamData.phase === 'critical'
                            ? '0 0 30px rgba(239, 68, 68, 0.12), inset 0 0 60px rgba(239, 68, 68, 0.04)'
                            : kParamData.phase === 'approaching'
                            ? '0 0 25px rgba(245, 158, 11, 0.08), inset 0 0 50px rgba(245, 158, 11, 0.03)'
                            : '0 0 20px rgba(16, 185, 129, 0.06), inset 0 0 40px rgba(16, 185, 129, 0.02)',
                        }}
                        onClick={() => setShowOrbitalInfo(true)}
                      >
                        <KOrbitalViz
                          kValue={kParamData.k_value ?? 0}
                          phase={kParamData.phase || 'stable'}
                          history={kParamHistory}
                        />
                        {/* Hover hint */}
                        <div className="absolute bottom-2 right-3 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                          <Eye className="w-3 h-3 text-white/30" />
                          <span className="text-[8px] text-white/30 tracking-wide uppercase">Click for field guide</span>
                        </div>
                      </motion.div>
                      <AnimatePresence>
                        {showOrbitalInfo && (
                          <KOrbitalInfoPanel
                            phase={kParamData.phase || 'stable'}
                            kValue={kParamData.k_value ?? 0}
                            show={showOrbitalInfo}
                            onClose={() => setShowOrbitalInfo(false)}
                          />
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Main K-Value Display */}
                    <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: `1px solid ${kParamData.phase === 'critical' ? 'rgba(239, 68, 68, 0.4)' : kParamData.phase === 'approaching' ? 'rgba(245, 158, 11, 0.4)' : 'rgba(16, 185, 129, 0.4)'}` }}>
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <div className="text-3xl font-bold font-mono" style={{ color: kParamData.phase === 'critical' ? '#ef4444' : kParamData.phase === 'approaching' ? '#f59e0b' : '#10b981' }}>
                            {(kParamData.k_value ?? 0).toFixed(4)}
                          </div>
                          <div className="text-[10px] text-amber-200/40 mt-1">Current K-Value</div>
                        </div>
                        <div className="text-right">
                          <span className={`inline-block px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider ${
                            kParamData.phase === 'critical' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                            kParamData.phase === 'approaching' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          }`}>
                            {kParamData.phase || 'stable'}
                          </span>
                          <div className="text-[10px] text-amber-200/30 mt-1.5">
                            Round #{kParamData.rounds_computed ?? 0}
                          </div>
                        </div>
                      </div>

                      {/* Phase Scale Bar */}
                      <div className="relative h-2 rounded-full bg-slate-700/50 overflow-hidden mt-2">
                        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-500" style={{
                          width: `${Math.min(100, ((kParamData.k_value ?? 0) / 15) * 100)}%`,
                          background: kParamData.phase === 'critical' ? 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)' :
                            kParamData.phase === 'approaching' ? 'linear-gradient(90deg, #10b981, #f59e0b)' :
                            '#10b981',
                        }} />
                        {/* Phase markers */}
                        <div className="absolute inset-y-0 left-[33.3%] w-px bg-amber-500/50" title="K=5 (Approaching)" />
                        <div className="absolute inset-y-0 left-[66.6%] w-px bg-red-500/50" title="K=10 (Critical)" />
                      </div>
                      <div className="flex justify-between text-[9px] text-amber-200/20 mt-1 px-1">
                        <span>0 (Stable)</span>
                        <span>5 (Approaching)</span>
                        <span>10 (Critical)</span>
                        <span>15+</span>
                      </div>
                    </div>

                    {/* Tuned Parameters Grid */}
                    <div className="grid grid-cols-3 gap-3">
                      <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                        <div className="text-[10px] text-amber-200/40 mb-1">Max Solutions/Block</div>
                        <div className="text-lg font-bold font-mono text-cyan-300">{kParamData.max_solutions_per_block ?? 250}</div>
                        <div className="text-[9px] text-amber-200/20 mt-0.5">Default: 250</div>
                      </div>
                      <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                        <div className="text-[10px] text-amber-200/40 mb-1">VDF Multiplier</div>
                        <div className="text-lg font-bold font-mono text-cyan-300">{(kParamData.vdf_multiplier ?? 1.0).toFixed(2)}x</div>
                        <div className="text-[9px] text-amber-200/20 mt-0.5">Default: 1.00x</div>
                      </div>
                      <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                        <div className="text-[10px] text-amber-200/40 mb-1">Challenge Expiry</div>
                        <div className="text-lg font-bold font-mono text-cyan-300">{kParamData.challenge_expiry_secs ?? 120}s</div>
                        <div className="text-[9px] text-amber-200/20 mt-0.5">Default: 120s</div>
                      </div>
                    </div>

                    {/* Phase Tuning Reference Table */}
                    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                      <table className="w-full text-xs">
                        <thead>
                          <tr style={{ background: 'rgba(15, 23, 42, 0.8)' }}>
                            <th className="text-left px-3 py-2 text-amber-200/50 font-medium">Parameter</th>
                            <th className="text-center px-3 py-2 text-emerald-400/70 font-medium">Stable (K&lt;5)</th>
                            <th className="text-center px-3 py-2 text-amber-400/70 font-medium">Approaching (5-10)</th>
                            <th className="text-center px-3 py-2 text-red-400/70 font-medium">Critical (K&gt;10)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { param: 'max_solutions', stable: '250', approaching: '150', critical: '50' },
                            { param: 'VDF multiplier', stable: '1.0x', approaching: '1.25x', critical: '1.5x' },
                            { param: 'Challenge expiry', stable: '120s', approaching: '90s', critical: '60s' },
                          ].map((row, i) => (
                            <tr key={row.param} style={{ background: i % 2 === 0 ? 'rgba(15, 23, 42, 0.4)' : 'rgba(15, 23, 42, 0.6)' }}>
                              <td className="px-3 py-1.5 text-amber-200/60 font-mono">{row.param}</td>
                              <td className={`px-3 py-1.5 text-center font-mono ${kParamData.phase === 'stable' ? 'text-emerald-300 font-bold' : 'text-emerald-300/40'}`}>{row.stable}</td>
                              <td className={`px-3 py-1.5 text-center font-mono ${kParamData.phase === 'approaching' ? 'text-amber-300 font-bold' : 'text-amber-300/40'}`}>{row.approaching}</td>
                              <td className={`px-3 py-1.5 text-center font-mono ${kParamData.phase === 'critical' ? 'text-red-300 font-bold' : 'text-red-300/40'}`}>{row.critical}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* K-Value History (sparkline-like) */}
                    {kParamHistory.length > 1 && (
                      <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                        <div className="text-[10px] text-amber-200/40 mb-2">K-Value History (last {kParamHistory.length} readings)</div>
                        <div className="flex items-end gap-px h-12">
                          {kParamHistory.map((point, i) => {
                            const maxK = Math.max(...kParamHistory.map(p => p.k), 1);
                            const heightPct = Math.max(2, (point.k / maxK) * 100);
                            return (
                              <div
                                key={i}
                                className="flex-1 rounded-t-sm transition-all duration-300"
                                style={{
                                  height: `${heightPct}%`,
                                  background: point.phase === 'critical' ? '#ef4444' : point.phase === 'approaching' ? '#f59e0b' : '#10b981',
                                  opacity: 0.4 + (i / kParamHistory.length) * 0.6,
                                }}
                                title={`K=${point.k.toFixed(4)} (${point.phase})`}
                              />
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* zk-STARK Proof Section */}
                    {kParamData.zk_commitment && (
                      <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                        <div className="flex items-center gap-2 mb-2">
                          <Lock className="w-3.5 h-3.5 text-purple-400" />
                          <span className="text-[10px] font-medium text-purple-200">zk-STARK Phase Proof</span>
                          {kParamData.zk_phase_proof?.verified && (
                            <span className="ml-auto flex items-center gap-1 text-[9px] text-emerald-400">
                              <CheckCircle className="w-3 h-3" /> Verified
                            </span>
                          )}
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="text-[9px] text-amber-200/30 w-20 shrink-0">Commitment</span>
                            <span className="text-[9px] font-mono text-purple-300/60 truncate">{kParamData.zk_commitment}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[9px] text-amber-200/30 w-20 shrink-0">Challenge</span>
                            <span className="text-[9px] font-mono text-purple-300/60 truncate">{kParamData.zk_phase_proof?.challenge || '—'}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[9px] text-amber-200/30 w-20 shrink-0">Response</span>
                            <span className="text-[9px] font-mono text-purple-300/60 truncate">{kParamData.zk_phase_proof?.response || '—'}</span>
                          </div>
                        </div>
                        <div className="text-[8px] text-amber-200/20 mt-2">
                          Fiat-Shamir non-interactive proof: phase membership verified without revealing exact K
                        </div>
                      </div>
                    )}

                    {/* Metric Inputs — Live Breakdown */}
                    <div className="rounded-xl p-3" style={{ background: 'rgba(15, 23, 42, 0.4)', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
                      <div className="text-[10px] text-amber-200/40 mb-2.5 font-medium">How K is Computed — Live Breakdown</div>
                      <div className="grid grid-cols-2 gap-4 text-[10px]">
                        {/* ΔH column */}
                        <div>
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-cyan-300/80 font-semibold">Energy Variance (ΔH)</span>
                            <span className="text-xs font-mono font-bold text-cyan-400">{(kParamData.delta_h ?? 0).toFixed(4)}</span>
                          </div>
                          {[
                            { label: 'Mining rejection ratio', key: 'rejection_ratio', color: '#f59e0b' },
                            { label: 'Traffic asymmetry (in/out)', key: 'traffic_asymmetry', color: '#06b6d4' },
                            { label: 'Peer churn rate', key: 'peer_churn', color: '#a855f7' },
                          ].map(item => {
                            const val = kParamData[item.key] ?? 0;
                            const barW = Math.min(val * 100, 100);
                            return (
                              <div key={item.key} className="mb-1">
                                <div className="flex items-center justify-between mb-0.5">
                                  <span className="text-amber-200/40">{item.label}</span>
                                  <span className="font-mono text-[9px]" style={{ color: item.color }}>{val.toFixed(4)}</span>
                                </div>
                                <div className="h-1 rounded-full bg-slate-700/50 overflow-hidden">
                                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.max(barW, 1)}%`, background: item.color }} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        {/* Δs column */}
                        <div>
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-cyan-300/80 font-semibold">Entropy Variance (Δs)</span>
                            <span className="text-xs font-mono font-bold text-cyan-400">{(kParamData.delta_s ?? 0).toFixed(4)}</span>
                          </div>
                          {[
                            { label: 'Sync divergence (local vs network)', key: 'sync_divergence', color: '#3b82f6' },
                            { label: 'Block rate deviation', key: 'block_rate_deviation', color: '#10b981' },
                          ].map(item => {
                            const val = kParamData[item.key] ?? 0;
                            const barW = Math.min(val * 100, 100);
                            return (
                              <div key={item.key} className="mb-1">
                                <div className="flex items-center justify-between mb-0.5">
                                  <span className="text-amber-200/40">{item.label}</span>
                                  <span className="font-mono text-[9px]" style={{ color: item.color }}>{val.toFixed(4)}</span>
                                </div>
                                <div className="h-1 rounded-full bg-slate-700/50 overflow-hidden">
                                  <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.max(barW, 1)}%`, background: item.color }} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                      {/* Formula bar */}
                      <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-white/5 text-[9px]">
                        <span className="text-amber-200/20 font-mono">K = 2π √(ΔH · Δs · ℏ) / τ</span>
                        <div className="flex items-center gap-2 text-amber-200/20">
                          <span>τ = 60s</span>
                          <span>·</span>
                          <span>ℏ = 1.0</span>
                          <span>·</span>
                          <span>Every 60s</span>
                          {kParamData.last_computed_at > 0 && (<>
                            <span>·</span>
                            <span className="text-cyan-400/40">Last: {new Date(kParamData.last_computed_at * 1000).toLocaleTimeString()}</span>
                          </>)}
                        </div>
                      </div>
                    </div>
                  </>)}
                </div>
              </>)}

              {activeTab === 'mining' && (<>
                <div className="space-y-4">
                  {/* Current Status */}
                  <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="w-4 h-4 text-amber-400" />
                      <span className="text-sm font-medium text-emerald-200">Current Forced Mode</span>
                      <button onClick={fetchMiningModeStatus} className="ml-auto p-1 rounded hover:bg-white/10">
                        <RefreshCw className="w-3 h-3 text-emerald-300/50" />
                      </button>
                    </div>
                    {miningModeStatus ? (
                      <div className="flex items-center gap-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          miningModeStatus.forced_mode === 'solo' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' :
                          miningModeStatus.forced_mode === 'pool' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' :
                          'bg-slate-500/20 text-slate-300 border border-slate-500/30'
                        }`}>
                          {miningModeStatus.forced_mode === 'none' ? 'No Override' : miningModeStatus.forced_mode.toUpperCase()}
                        </span>
                        {miningModeStatus.pool_url && (
                          <span className="text-xs text-amber-200/50 truncate max-w-[200px]">{miningModeStatus.pool_url}</span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-amber-200/30">Click refresh to load status</span>
                    )}
                  </div>

                  {/* Pool URL Input */}
                  <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                    <label className="text-xs text-amber-200/60 mb-1 block">Pool URL (for pool mode)</label>
                    <input
                      type="text"
                      value={miningPoolUrlInput}
                      onChange={e => setMiningPoolUrlInput(e.target.value)}
                      placeholder="stratum+tcp://quillon.xyz:3333"
                      className="w-full px-3 py-2 rounded-lg text-xs bg-slate-900/60 border border-slate-600/30 text-emerald-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
                    />
                  </div>

                  {/* Action Buttons */}
                  <div className="grid grid-cols-3 gap-2">
                    <motion.button
                      onClick={() => triggerMiningModeSwitch('solo')}
                      disabled={miningModeLoading}
                      className="flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl text-xs font-medium transition-all disabled:opacity-50"
                      style={{ background: 'rgba(59, 130, 246, 0.15)', border: '1px solid rgba(59, 130, 246, 0.3)', color: 'rgb(147, 197, 253)' }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Cpu className="w-3.5 h-3.5" />
                      Switch to Solo
                    </motion.button>

                    <motion.button
                      onClick={() => triggerMiningModeSwitch('pool')}
                      disabled={miningModeLoading}
                      className="flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl text-xs font-medium transition-all disabled:opacity-50"
                      style={{ background: 'rgba(168, 85, 247, 0.15)', border: '1px solid rgba(168, 85, 247, 0.3)', color: 'rgb(196, 181, 253)' }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <Users className="w-3.5 h-3.5" />
                      Switch to Pool
                    </motion.button>

                    <motion.button
                      onClick={() => triggerMiningModeSwitch('clear')}
                      disabled={miningModeLoading}
                      className="flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl text-xs font-medium transition-all disabled:opacity-50"
                      style={{ background: 'rgba(100, 116, 139, 0.15)', border: '1px solid rgba(100, 116, 139, 0.3)', color: 'rgb(203, 213, 225)' }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <X className="w-3.5 h-3.5" />
                      Clear Override
                    </motion.button>
                  </div>

                  {/* Status Message */}
                  {miningModeMsg && (
                    <div className={`rounded-lg p-3 text-xs ${
                      miningModeMsg.type === 'success' ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' : 'bg-red-500/10 text-red-300 border border-red-500/20'
                    }`}>
                      {miningModeMsg.type === 'success' ? <CheckCircle className="w-3.5 h-3.5 inline mr-1.5" /> : <XCircle className="w-3.5 h-3.5 inline mr-1.5" />}
                      {miningModeMsg.text}
                    </div>
                  )}

                  {/* Info */}
                  <div className="rounded-xl p-3 text-xs text-amber-200/40" style={{ background: 'rgba(15, 23, 42, 0.4)', border: '1px solid rgba(148, 163, 184, 0.05)' }}>
                    <p className="mb-1"><strong>How it works:</strong></p>
                    <p>Broadcasts an SSE event to all connected miners and piggybacks on the next challenge response. Miners gracefully stop current batch and restart in the new mode. Old miners (pre-v9.1.4) ignore unknown fields.</p>
                  </div>
                </div>
              </>)}

              {/* ═══════ v9.5.0: STARSHIP COMPUTE TAB ═══════ */}
              {activeTab === 'compute' && (<>
                <div className="space-y-4">
                  {/* Compute Mode */}
                  <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                    <div className="flex items-center gap-2 mb-3">
                      <Rocket className="w-4 h-4 text-amber-400" />
                      <span className="text-sm font-medium text-emerald-200">Compute Orchestrator</span>
                      <button onClick={fetchComputeData} className="ml-auto p-1 rounded hover:bg-white/10">
                        <RefreshCw className={`w-3 h-3 text-emerald-300/50 ${computeLoading ? 'animate-spin' : ''}`} />
                      </button>
                    </div>
                    {computeData ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-3">
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                            computeData.mode === 'Nuke' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                            computeData.mode === 'Full' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
                            computeData.mode === 'Eco' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' :
                            'bg-slate-500/20 text-slate-300 border border-slate-500/30'
                          }`}>
                            MODE: {typeof computeData.mode === 'string' ? computeData.mode.toUpperCase() : 'UNKNOWN'}
                          </span>
                          {computeData.trainer_active && (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/30 animate-pulse">
                              TRAINER ACTIVE
                            </span>
                          )}
                          {computeData.performance_boost_pct > 0 && (
                            <span className="text-xs text-amber-300">+{computeData.performance_boost_pct.toFixed(0)}% boost</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <span className="text-xs text-amber-200/30">Click refresh to load status</span>
                    )}
                  </div>

                  {/* Resource Utilization */}
                  {computeData?.resources && (
                    <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-4 h-4 text-blue-400" />
                        <span className="text-sm font-medium text-emerald-200">Resource Utilization</span>
                      </div>
                      <div className="space-y-2">
                        {[
                          { label: 'CPU', value: computeData.resources.cpu_total, color: 'emerald' },
                          { label: 'GPU', value: computeData.resources.gpu_utilization, color: 'purple' },
                          { label: 'RAM', value: computeData.resources.ram_total > 0 ? (computeData.resources.ram_used / computeData.resources.ram_total * 100) : 0, color: 'blue' },
                        ].map(r => (
                          <div key={r.label} className="flex items-center gap-2">
                            <span className="text-[10px] text-amber-200/60 w-8">{r.label}</span>
                            <div className="flex-1 h-2 rounded-full bg-slate-800/60 overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-500 ${
                                  r.color === 'emerald' ? 'bg-emerald-500' : r.color === 'purple' ? 'bg-purple-500' : 'bg-blue-500'
                                }`}
                                style={{ width: `${Math.min(r.value, 100)}%` }}
                              />
                            </div>
                            <span className="text-[10px] text-amber-200/80 w-10 text-right">{r.value.toFixed(1)}%</span>
                          </div>
                        ))}
                        {/* Network stats */}
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-[10px] text-amber-200/60 w-8">NET</span>
                          <span className="text-[10px] text-amber-200/60">
                            ↑{((computeData.resources.net_tx_bps || 0) / 1e6).toFixed(1)} MB/s
                            ↓{((computeData.resources.net_rx_bps || 0) / 1e6).toFixed(1)} MB/s
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-amber-200/60 w-8">DISK</span>
                          <span className="text-[10px] text-amber-200/60">
                            {((computeData.resources.disk_io_bps || 0) / 1e6).toFixed(1)} MB/s
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 8-Layer Priority Scheduler */}
                  {computeData?.layers && computeData.layers.length > 0 && (
                    <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                      <div className="flex items-center gap-2 mb-3">
                        <Layers className="w-4 h-4 text-purple-400" />
                        <span className="text-sm font-medium text-emerald-200">8-Layer Compute Scheduler</span>
                      </div>
                      <div className="space-y-1">
                        {computeData.layers.map(([name, stats]: [string, any], i: number) => (
                          <div key={name} className="flex items-center gap-2 text-[10px]">
                            <span className={`w-3 h-3 rounded-sm ${stats.active_since_ms > 0 ? 'bg-emerald-500' : 'bg-slate-700'}`} />
                            <span className="text-amber-200/80 w-20 truncate">L{i}: {name}</span>
                            <span className="text-amber-200/40 w-14">{stats.cores_assigned} cores</span>
                            <span className="text-amber-200/40 w-14">{stats.tasks_completed} done</span>
                            <span className="text-amber-200/40">{stats.tasks_pending} pending</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Trainer Cheats */}
                  {computeData?.trainer_cheats && (
                    <div className="rounded-xl p-4" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.1)' }}>
                      <div className="flex items-center gap-2 mb-3">
                        <Zap className="w-4 h-4 text-red-400" />
                        <span className="text-sm font-medium text-emerald-200">Trainer Cheats</span>
                        {computeData.trainer_active && (
                          <span className="text-[10px] text-red-300 animate-pulse ml-auto">ALL CHEATS ACTIVE</span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 gap-1">
                        {['F1:INFINITE_CORES', 'F2:GOD_MODE_MEMORY', 'F3:SPEED_HACK_x100', 'F4:WALL_HACK',
                          'F5:AIM_BOT', 'F6:NO_CLIP', 'F7:INFINITE_AMMO', 'F8:RAPID_FIRE',
                          'F9:TELEPORT', 'F10:PRESTIGE_MODE'].map(cheat => {
                          const active = computeData.trainer_cheats.includes(cheat);
                          return (
                            <div key={cheat} className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] ${
                              active ? 'bg-red-500/10 text-red-300 border border-red-500/20' : 'bg-slate-800/40 text-slate-500'
                            }`}>
                              <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-red-400' : 'bg-slate-600'}`} />
                              {cheat.replace('_', ' ')}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </>)}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
