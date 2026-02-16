/**
 * v5.7.0: Deploy Control Panel - CCC Convergence-Aware Deployment Management
 *
 * Integrates the K-Kristensen Convergence Readiness framework from the
 * "Cosmic Arcology Mission" paper into the 4-server HA deployment pipeline.
 *
 * Cosmic phases map to deployment stages:
 *   Isolation → Alpha + Delta deploying in parallel
 *   Convergence → Gamma verifying, syncing with Beta
 *   Aeon Transition → Beta deploying (conformal boundary crossing)
 *   Harmony → All 4 servers unified, same version, synced
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
  Database, TrendingUp, MonitorSmartphone, Timer, DollarSign, Settings, Save
} from 'lucide-react';
import { getConnectionInfo } from '../services/api';

const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';

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
  height_delta: number;
  versions_match: boolean;
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

/** K-metrics breakdown bar for a single node */
function KMetricsBar({ metrics }: { metrics: NodeKMetrics }) {
  const factors = [
    { key: 'G', value: metrics.genetic_stability, label: 'Genetic', exp: 0.25 },
    { key: 'Q', value: metrics.quantum_coherence, label: 'Coherence', exp: 0.20 },
    { key: 'T', value: metrics.thermodynamic_efficiency, label: 'Thermo', exp: 0.20 },
    { key: 'I', value: metrics.information_density, label: 'Info', exp: 0.15 },
    { key: 'R', value: metrics.network_resilience, label: 'Resilience', exp: 0.20 },
  ];

  return (
    <div className="flex items-center gap-1">
      {factors.map(f => {
        const color = f.value > 0.8 ? 'bg-emerald-500' : f.value > 0.5 ? 'bg-amber-500' : 'bg-red-500';
        return (
          <div key={f.key} className="flex-1 group relative">
            <div className="h-1.5 rounded-full bg-slate-700/50 overflow-hidden">
              <div className={`h-full rounded-full ${color} transition-all duration-700`}
                style={{ width: `${f.value * 100}%` }} />
            </div>
            <div className="opacity-0 group-hover:opacity-100 absolute -top-8 left-1/2 -translate-x-1/2
              bg-slate-800 border border-slate-600 rounded px-1.5 py-0.5 text-[9px] text-amber-200/80 whitespace-nowrap z-10 pointer-events-none transition-opacity">
              {f.key}={f.value.toFixed(2)} (^{f.exp})
            </div>
          </div>
        );
      })}
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

function ServerCard({ node, isActive, role, syncMetrics }: { node: NodeStatus; isActive: boolean; role: 'canary' | 'primary' | 'backup' | 'bootstrap'; syncMetrics?: SyncMetrics }) {
  const roleConfig = {
    canary: { label: 'CANARY', color: 'text-purple-300', bg: 'bg-purple-500/20', border: 'border-purple-400/40' },
    primary: { label: 'PRIMARY', color: 'text-emerald-300', bg: 'bg-emerald-500/20', border: 'border-emerald-400/40' },
    bootstrap: { label: 'BOOTSTRAP', color: 'text-cyan-300', bg: 'bg-cyan-500/20', border: 'border-cyan-400/40' },
    backup: { label: 'BACKUP', color: 'text-blue-300', bg: 'bg-blue-500/20', border: 'border-blue-400/40' },
  }[role];

  return (
    <div className={`rounded-xl border p-3 relative ${
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
          <StatusIcon status={node.status} />
          <span className={`text-xs font-medium ${
            node.status === 'ready' ? 'text-emerald-400' :
            node.status === 'syncing' ? 'text-amber-400' :
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
              {syncMetrics && syncMetrics.speed > 0 ? (
                <><TrendingUp className="w-2.5 h-2.5 text-emerald-400" />{syncMetrics.speed.toLocaleString()} blk/s</>
              ) : (
                'Sync'
              )}
            </span>
            <span className="flex items-center gap-1.5">
              {syncMetrics && syncMetrics.eta_secs > 0 && (
                <span className="text-amber-300/70">ETA: {formatEta(syncMetrics.eta_secs)}</span>
              )}
              {syncMetrics && syncMetrics.eta_secs === -1 && (
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
  const prevHeightsRef = useRef<Record<string, { height: number; ts: number }>>({});
  const eventSourceRef = useRef<EventSource | null>(null);

  // Check if current wallet is master
  const walletAddress = localStorage.getItem('walletAddress') || '';
  const cleanWallet = walletAddress.replace('qnk', '').replace('qug', '');
  const isMaster = cleanWallet === MASTER_WALLET;

  // Listen for open event from TopBar admin button
  useEffect(() => {
    const handler = () => setIsOpen(true);
    window.addEventListener('open-deploy-panel', handler);
    return () => window.removeEventListener('open-deploy-panel', handler);
  }, []);

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
  const fetchStatus = useCallback(async () => {
    if (!isMaster) return;
    setLoading(true);
    setError(null);
    const headers = {
      'X-Wallet-Auth': walletAddress,
      'Authorization': `Bearer ${walletAddress}`,
    };
    try {
      const [statusResp, convResp, devFeeResp] = await Promise.all([
        fetch('/api/v1/admin/deploy/status', { headers }),
        fetch('/api/v1/admin/deploy/convergence', { headers }).catch(() => null),
        fetch('/api/v1/admin/dev-fee', { headers }).catch(() => null),
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
    } catch (e: any) {
      setError(e.message || 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, [isMaster, walletAddress]);

  // Auto-refresh when panel opens
  useEffect(() => {
    if (isOpen && isMaster) {
      fetchStatus();
      setConnInfo(getConnectionInfo());
      const interval = setInterval(() => {
        fetchStatus();
        setConnInfo(getConnectionInfo());
      }, 15000);
      return () => clearInterval(interval);
    }
  }, [isOpen, isMaster, fetchStatus]);

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
    if (!confirm('Deploy to all 4 servers?\n\nPipeline: Alpha+Delta (parallel) -> Gamma (verify) -> Beta (primary)')) return;
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

  if (!isMaster) return null;

  const allPassed = verifyEvents.length > 0 &&
    verifyEvents.some(e => e.step === 'RESULT' && e.status === 'passed');
  const anyFailed = verifyEvents.some(e => e.status === 'failed');

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
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
            className="fixed top-4 left-1/2 transform -translate-x-1/2 w-[820px] max-w-[95vw] max-h-[90vh] overflow-y-auto rounded-2xl z-[99999]"
            style={{
              background: 'linear-gradient(135deg, rgba(15, 10, 35, 0.98) 0%, rgba(30, 20, 55, 0.98) 100%)',
              border: '2px solid rgba(16, 185, 129, 0.3)',
              boxShadow: '0 25px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(16, 185, 129, 0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-5 border-b border-emerald-500/20">
              <div className="flex items-center gap-3">
                <Shield className="w-6 h-6 text-emerald-400" />
                <div>
                  <h2 className="text-lg font-bold text-emerald-50">Node Admin</h2>
                  <p className="text-xs text-emerald-300/60">Deploy Control Panel</p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
              >
                <X className="w-5 h-5 text-emerald-300/60" />
              </button>
            </div>

            <div className="p-5 space-y-4">
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
                  <div className="grid grid-cols-4 gap-2">
                    <ServerCard
                      node={deployStatus.alpha}
                      isActive={false}
                      role="canary"
                      syncMetrics={syncMetricsMap.alpha}
                    />
                    <ServerCard
                      node={deployStatus.beta}
                      isActive={connInfo.isPrimary}
                      role="primary"
                      syncMetrics={syncMetricsMap.beta}
                    />
                    <ServerCard
                      node={deployStatus.gamma}
                      isActive={!connInfo.isPrimary}
                      role="backup"
                      syncMetrics={syncMetricsMap.gamma}
                    />
                    <ServerCard
                      node={deployStatus.delta}
                      isActive={false}
                      role="bootstrap"
                      syncMetrics={syncMetricsMap.delta}
                    />
                  </div>

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
                          <span className="text-lg">{getPhaseInfo(convergence.cosmic_phase).icon}</span>
                          <div>
                            <div className="flex items-center gap-1.5">
                              <span className={`text-xs font-bold uppercase tracking-wider ${getPhaseInfo(convergence.cosmic_phase).color}`}>
                                {getPhaseInfo(convergence.cosmic_phase).name}
                              </span>
                              <span className="text-[9px] text-amber-200/40">Cosmic Phase</span>
                            </div>
                            <div className="text-[10px] text-amber-200/50 mt-0.5">
                              K-Kristensen Convergence Readiness
                            </div>
                          </div>
                        </div>

                        {/* Collective K gauge */}
                        <div className="flex items-center gap-3">
                          <KGauge value={convergence.collective_k} label="Collective K" size="lg" />
                          <div className="flex flex-col items-end gap-0.5">
                            <div className={`text-[10px] font-bold ${getOutcomeInfo(convergence.predicted_outcome).color}`}>
                              {getOutcomeInfo(convergence.predicted_outcome).name}
                            </div>
                            <div className="text-[9px] text-amber-200/40">
                              {getOutcomeInfo(convergence.predicted_outcome).desc}
                            </div>
                            {convergence.convergence_safe ? (
                              <div className="flex items-center gap-0.5 mt-0.5">
                                <CheckCircle className="w-2.5 h-2.5 text-emerald-400" />
                                <span className="text-[9px] text-emerald-400 font-medium">Safe to deploy</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-0.5 mt-0.5">
                                <AlertTriangle className="w-2.5 h-2.5 text-amber-400" />
                                <span className="text-[9px] text-amber-400 font-medium">Verify first</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Per-Node K-Metrics */}
                      <div className="space-y-1.5 mb-2">
                        {convergence.nodes.filter(n => n.k_parameter > 0).map(node => (
                          <div key={node.name} className="flex items-center gap-2">
                            <span className="text-[10px] text-amber-200/60 w-16 truncate">{node.name.replace('Server ', '')}</span>
                            <div className="flex-1">
                              <KMetricsBar metrics={node} />
                            </div>
                            <span className={`text-[10px] font-mono font-bold w-8 text-right ${
                              node.k_parameter > 0.9 ? 'text-emerald-400' :
                              node.k_parameter > 0.7 ? 'text-blue-400' :
                              node.k_parameter > 0.5 ? 'text-amber-400' : 'text-red-400'
                            }`}>
                              {node.k_parameter.toFixed(2)}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* K-Formula legend */}
                      <div className="flex items-center justify-center gap-2 text-[8px] text-amber-200/30 mb-2">
                        <span>k = G<sup>.25</sup></span>
                        <span>&times;</span>
                        <span>Q<sup>.20</sup></span>
                        <span>&times;</span>
                        <span>T<sup>.20</sup></span>
                        <span>&times;</span>
                        <span>I<sup>.15</sup></span>
                        <span>&times;</span>
                        <span>R<sup>.20</sup></span>
                      </div>

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
                        {(deployStatus.alpha.online ? deployStatus.alpha.peers : 0) + (deployStatus.beta.online ? deployStatus.beta.peers : 0) + (deployStatus.gamma.online ? deployStatus.gamma.peers : 0) + (deployStatus.delta?.online ? deployStatus.delta.peers : 0)}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-2 text-center">
                      <div className="text-[10px] text-amber-200/50 uppercase tracking-wider mb-1">Servers</div>
                      <div className="text-sm font-bold text-emerald-400">
                        {(deployStatus.alpha.online ? 1 : 0) + (deployStatus.beta.online ? 1 : 0) + (deployStatus.gamma.online ? 1 : 0) + (deployStatus.delta?.online ? 1 : 0)}/4
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

              {/* Action Buttons */}
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
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
