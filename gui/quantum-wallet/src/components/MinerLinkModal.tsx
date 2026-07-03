import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Pickaxe, Cpu, Activity, Gauge, Play, Pause, Minus, Plus,
  Wifi, WifiOff, Clock, Zap, Hash, Monitor, Settings, Link2, ChevronRight
} from 'lucide-react';
import type { MinerInfo, MinerCommandAction, PendingCommand, UseMinerLinkReturn, GpuDeviceInfo } from '../hooks/useMinerLink';

interface MinerLinkModalProps {
  isOpen: boolean;
  onClose: () => void;
  minerLink: UseMinerLinkReturn;
}

type TabId = 'overview' | 'details' | 'controls' | 'connection';

function formatHashrate(hs: number): string {
  if (hs >= 1e9) return `${(hs / 1e9).toFixed(2)} GH/s`;
  if (hs >= 1e6) return `${(hs / 1e6).toFixed(2)} MH/s`;
  if (hs >= 1e3) return `${(hs / 1e3).toFixed(2)} KH/s`;
  return `${hs.toFixed(0)} H/s`;
}

function formatUptime(secs: number): string {
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${secs % 60}s`;
}

// Tiny sparkline SVG from an array of numbers
function Sparkline({ data, color = '#a78bfa', width = 120, height = 32 }: { data: number[]; color?: string; width?: number; height?: number }) {
  if (data.length < 2) return <div style={{ width, height }} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={width} height={height} className="inline-block">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

const MinerLinkModal: React.FC<MinerLinkModalProps> = ({ isOpen, onClose, minerLink }) => {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [selectedMinerId, setSelectedMinerId] = useState<string | null>(null);

  const { isConnected, miners, totalHashrate, totalSolutions, sendCommand, pendingCommands } = minerLink;

  const selectedMiner = useMemo(
    () => miners.find(m => m.minerId === selectedMinerId) ?? miners[0] ?? null,
    [miners, selectedMinerId]
  );

  if (!isOpen) return null;

  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Activity className="w-4 h-4" /> },
    { id: 'details', label: 'Miner Details', icon: <Monitor className="w-4 h-4" /> },
    { id: 'controls', label: 'Controls', icon: <Settings className="w-4 h-4" /> },
    { id: 'connection', label: 'Connection', icon: <Link2 className="w-4 h-4" /> },
  ];

  const walletAddress = localStorage.getItem('walletAddress') || '';
  const currentServerUrl = localStorage.getItem('apiBaseURL') || 'https://quillon.xyz';
  const miningCommand = `# CPU mining:\n./q-miner --mode solo --wallet ${walletAddress} --threads 4 --intensity 7 --server ${currentServerUrl}\n\n# GPU mining:\n./q-miner --mode solo --wallet ${walletAddress} --gpu --server ${currentServerUrl}`;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 overflow-y-auto"
        style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(12px)' }}
        onClick={onClose}
      >
        <div className="flex min-h-full items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.92, y: 24 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.92, y: 24 }}
          transition={{ type: 'spring', damping: 26, stiffness: 300 }}
          className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border"
          style={{
            background: 'linear-gradient(135deg, #0c0a1a 0%, #14102a 50%, #0a0816 100%)',
            borderColor: 'rgba(139, 92, 246, 0.2)',
            boxShadow: '0 0 80px rgba(139, 92, 246, 0.1)',
          }}
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b" style={{ borderColor: 'rgba(139, 92, 246, 0.12)' }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, #7c3aed, #a78bfa)' }}
                >
                  <Pickaxe className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Miner Link</h2>
                  <p className="text-sm text-gray-400">
                    {miners.length > 0
                      ? `${miners.length} miner${miners.length > 1 ? 's' : ''} connected — ${formatHashrate(totalHashrate)}`
                      : 'No miners connected'}
                  </p>
                </div>
                {/* Connection indicator */}
                <div className={`ml-2 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                  isConnected && miners.length > 0
                    ? 'bg-green-500/15 text-green-400'
                    : isConnected
                    ? 'bg-amber-500/15 text-amber-400'
                    : 'bg-red-500/15 text-red-400'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    isConnected && miners.length > 0 ? 'bg-green-400 animate-pulse' : isConnected ? 'bg-amber-400' : 'bg-red-400'
                  }`} />
                  {isConnected && miners.length > 0 ? 'Live' : isConnected ? 'Waiting' : 'Disconnected'}
                </div>
              </div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 mt-4 bg-white/5 rounded-lg p-1">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all flex-1 justify-center ${
                    activeTab === tab.id
                      ? 'bg-purple-500/20 text-purple-300 shadow-sm'
                      : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                  }`}
                >
                  {tab.icon}
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {activeTab === 'overview' && (
              <OverviewTab miners={miners} totalHashrate={totalHashrate} totalSolutions={totalSolutions} isConnected={isConnected} />
            )}
            {activeTab === 'details' && (
              <DetailsTab
                miners={miners}
                selectedMiner={selectedMiner}
                onSelectMiner={setSelectedMinerId}
              />
            )}
            {activeTab === 'controls' && (
              <ControlsTab
                selectedMiner={selectedMiner}
                miners={miners}
                onSelectMiner={setSelectedMinerId}
                sendCommand={sendCommand}
                pendingCommands={pendingCommands}
              />
            )}
            {activeTab === 'connection' && (
              <ConnectionTab isConnected={isConnected} miners={miners} miningCommand={miningCommand} />
            )}
          </div>
        </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

// ─── Overview Tab ────────────────────────────────────────────────────────────

function OverviewTab({ miners, totalHashrate, totalSolutions, isConnected }: {
  miners: MinerInfo[];
  totalHashrate: number;
  totalSolutions: number;
  isConnected: boolean;
}) {
  const activeMinerCount = miners.filter(m => m.isMining).length;

  return (
    <div className="space-y-6">
      {/* Status banner */}
      <div className={`p-4 rounded-xl border ${
        isConnected && miners.length > 0
          ? 'bg-green-500/5 border-green-500/20'
          : 'bg-amber-500/5 border-amber-500/20'
      }`}>
        <div className="flex items-center gap-3">
          {isConnected && miners.length > 0 ? (
            <>
              <Wifi className="w-5 h-5 text-green-400" />
              <span className="text-green-300 font-medium">
                Connected to your miner{miners.length > 1 ? 's' : ''} via Tor relay
              </span>
            </>
          ) : (
            <>
              <WifiOff className="w-5 h-5 text-amber-400" />
              <span className="text-amber-300 font-medium">
                {isConnected ? 'Relay connected — waiting for miner' : 'No miners connected'}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryCard icon={<Gauge className="w-5 h-5" />} label="Total Hashrate" value={formatHashrate(totalHashrate)} color="purple" />
        <SummaryCard icon={<Pickaxe className="w-5 h-5" />} label="Active Miners" value={`${activeMinerCount}`} color="green" />
        <SummaryCard icon={<Zap className="w-5 h-5" />} label="Solutions" value={`${totalSolutions}`} color="amber" />
        <SummaryCard icon={<Hash className="w-5 h-5" />} label="Total Hashes" value={formatLargeNumber(miners.reduce((s, m) => s + m.totalHashes, 0))} color="blue" />
      </div>

      {/* Per-miner cards */}
      {miners.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Connected Miners</h3>
          {miners.map(miner => (
            <div
              key={miner.minerId}
              className="p-4 rounded-xl border bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
              style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`w-2.5 h-2.5 rounded-full ${miner.isMining ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
                  <span className="text-white font-medium">
                    {miner.minerName || `Miner ${miner.minerId.slice(0, 8)}`}
                  </span>
                  <span className="text-xs text-gray-500 font-mono">{miner.minerId.slice(0, 12)}...</span>
                  {miner.gpuActive && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/20">GPU</span>
                  )}
                </div>
                <span className={`text-sm font-medium ${miner.isMining ? 'text-green-400' : 'text-gray-500'}`}>
                  {miner.isMining ? 'Mining' : 'Paused'}
                </span>
              </div>
              <div className="flex items-center gap-6">
                <div>
                  <span className="text-lg font-bold text-purple-300">{formatHashrate(miner.hashrate)}</span>
                </div>
                <Sparkline data={miner.hashrateHistory} color={miner.isMining ? '#a78bfa' : '#6b7280'} />
                <div className="ml-auto text-right">
                  <div className="text-xs text-gray-500">Uptime</div>
                  <div className="text-sm text-gray-300">{formatUptime(miner.uptimeSecs)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Details Tab ─────────────────────────────────────────────────────────────

function DetailsTab({ miners, selectedMiner, onSelectMiner }: {
  miners: MinerInfo[];
  selectedMiner: MinerInfo | null;
  onSelectMiner: (id: string) => void;
}) {
  if (miners.length === 0) {
    return <EmptyState message="No miners connected. Start your miner to see detailed stats." />;
  }

  const miner = selectedMiner || miners[0];

  const simdLabel = miner.hasAvx512 ? 'AVX-512' : miner.hasAvx2 ? 'AVX2' : 'SSE';
  const efficiency = miner.threadsActive > 0 ? miner.hashrate / miner.threadsActive : 0;
  const peakHashrate = miner.hashrateHistory.length > 0 ? Math.max(...miner.hashrateHistory) : 0;

  return (
    <div className="space-y-6">
      {/* Miner selector */}
      {miners.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {miners.map(m => (
            <button
              key={m.minerId}
              onClick={() => onSelectMiner(m.minerId)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                m.minerId === miner.minerId
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  : 'bg-white/5 text-gray-400 hover:bg-white/10 border border-transparent'
              }`}
            >
              {m.minerName || `Miner ${m.minerId.slice(0, 8)}`}
            </button>
          ))}
        </div>
      )}

      {/* GPU Hardware (shown when GPU miner is active) */}
      {miner.gpuActive && miner.gpuDevices.length > 0 && (
        <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(168, 85, 247, 0.15)' }}>
          <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
            <span className="text-base">🎮</span> GPU Hardware
          </h4>
          {miner.gpuDevices.map((gpu, i) => (
            <div key={i} className={`${i > 0 ? 'mt-3 pt-3 border-t border-white/5' : ''}`}>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">GPU</span>
                  <p className="text-white font-medium">{gpu.name}</p>
                </div>
                <div>
                  <span className="text-gray-500">Vendor</span>
                  <p className="text-white">{gpu.vendor}</p>
                </div>
                <div>
                  <span className="text-gray-500">Compute Units</span>
                  <p className="text-white">{gpu.compute_units.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-500">VRAM</span>
                  <p className="text-white">{gpu.memory_mb >= 1024 ? `${(gpu.memory_mb / 1024).toFixed(1)} GB` : `${gpu.memory_mb} MB`}</p>
                </div>
                <div>
                  <span className="text-gray-500">Clock</span>
                  <p className="text-white">{gpu.max_clock_mhz} MHz</p>
                </div>
                <div>
                  <span className="text-gray-500">API</span>
                  <p className="text-white">{gpu.api}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CPU Hardware */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4" /> {miner.gpuActive ? 'CPU (Hybrid Mode)' : 'Hardware'}
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">CPU</span>
            <p className="text-white">{miner.cpuVendor}</p>
          </div>
          <div>
            <span className="text-gray-500">SIMD</span>
            <p className="text-white">{simdLabel}</p>
          </div>
          <div>
            <span className="text-gray-500">Threads</span>
            <p className="text-white">{miner.threadsActive} / {miner.threadsTotal}</p>
          </div>
          <div>
            <span className="text-gray-500">Intensity</span>
            <p className="text-white">{miner.intensity} / 10</p>
          </div>
        </div>
      </div>

      {/* Performance */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
          <Gauge className="w-4 h-4" /> Performance
        </h4>
        {/* GPU hashrate row (if active) */}
        {miner.gpuActive && miner.gpuHashrate > 0 && (
          <div className="mb-4 p-3 rounded-lg bg-purple-500/5 border border-purple-500/10">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400 flex items-center gap-1.5"><span>🎮</span> GPU Hashrate</span>
              <span className="text-lg font-bold text-purple-300">{formatHashrate(miner.gpuHashrate)}</span>
            </div>
          </div>
        )}
        <div className="grid grid-cols-3 gap-4 text-sm mb-4">
          <div>
            <span className="text-gray-500">{miner.gpuActive ? 'Combined' : 'Current'}</span>
            <p className="text-lg font-bold text-purple-300">{formatHashrate(miner.hashrate)}</p>
          </div>
          <div>
            <span className="text-gray-500">Peak</span>
            <p className="text-lg font-bold text-amber-300">{formatHashrate(peakHashrate)}</p>
          </div>
          <div>
            <span className="text-gray-500">Average</span>
            <p className="text-lg font-bold text-blue-300">{formatHashrate(miner.avgHashrate5m)}</p>
          </div>
        </div>
        <div className="mb-2 text-xs text-gray-500">Hashrate (60s)</div>
        <Sparkline data={miner.hashrateHistory} width={600} height={48} />
        <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Efficiency</span>
            <p className="text-white">{miner.gpuActive && miner.gpuDevices.length > 0
              ? `${formatHashrate(miner.hashrate / Math.max(miner.gpuDevices.reduce((s, g) => s + g.compute_units, 0), 1))} / CU`
              : `${formatHashrate(efficiency)} / thread`
            }</p>
          </div>
          <div>
            <span className="text-gray-500">Solutions / hr</span>
            <p className="text-white">
              {miner.uptimeSecs > 0 ? ((miner.solutions / miner.uptimeSecs) * 3600).toFixed(1) : '0'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Controls Tab ────────────────────────────────────────────────────────────

function ControlsTab({ selectedMiner, miners, onSelectMiner, sendCommand, pendingCommands }: {
  selectedMiner: MinerInfo | null;
  miners: MinerInfo[];
  onSelectMiner: (id: string) => void;
  sendCommand: (minerId: string, cmd: MinerCommandAction) => void;
  pendingCommands: PendingCommand[];
}) {
  if (!selectedMiner || miners.length === 0) {
    return <EmptyState message="No miners connected. Start your miner to access controls." />;
  }

  const miner = selectedMiner;

  return (
    <div className="space-y-6">
      {/* Miner selector */}
      {miners.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {miners.map(m => (
            <button
              key={m.minerId}
              onClick={() => onSelectMiner(m.minerId)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                m.minerId === miner.minerId
                  ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  : 'bg-white/5 text-gray-400 hover:bg-white/10 border border-transparent'
              }`}
            >
              {m.minerName || `Miner ${m.minerId.slice(0, 8)}`}
            </button>
          ))}
        </div>
      )}

      {/* Pause / Resume */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-white font-medium">Mining Status</h4>
            <p className="text-sm text-gray-400 mt-1">
              {miner.isMining ? 'Your miner is actively hashing' : 'Mining is paused'}
            </p>
          </div>
          <button
            onClick={() => sendCommand(miner.minerId, { action: miner.isMining ? 'Pause' : 'Resume' })}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all ${
              miner.isMining
                ? 'bg-amber-500/15 text-amber-300 hover:bg-amber-500/25 border border-amber-500/20'
                : 'bg-green-500/15 text-green-300 hover:bg-green-500/25 border border-green-500/20'
            }`}
          >
            {miner.isMining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {miner.isMining ? 'Pause' : 'Resume'}
          </button>
        </div>
      </div>

      {/* Thread count */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-white font-medium mb-3">Thread Count</h4>
        <div className="flex items-center gap-4">
          <button
            onClick={() => sendCommand(miner.minerId, { action: 'SetThreads', count: Math.max(1, miner.threadsActive - 1) })}
            className="w-10 h-10 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center text-gray-300 transition-colors"
          >
            <Minus className="w-4 h-4" />
          </button>
          <div className="text-center flex-1">
            <span className="text-2xl font-bold text-white">{miner.threadsActive}</span>
            <span className="text-gray-500 text-sm"> / {miner.threadsTotal}</span>
          </div>
          <button
            onClick={() => sendCommand(miner.minerId, { action: 'SetThreads', count: Math.min(miner.threadsTotal, miner.threadsActive + 1) })}
            className="w-10 h-10 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center text-gray-300 transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        {/* Visual bar */}
        <div className="mt-3 w-full h-3 rounded-full bg-white/5 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${(miner.threadsActive / Math.max(miner.threadsTotal, 1)) * 100}%`,
              background: 'linear-gradient(90deg, #7c3aed, #a78bfa)',
            }}
          />
        </div>
      </div>

      {/* Intensity slider */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-white font-medium mb-3">Mining Intensity</h4>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-500 w-6">1</span>
          <input
            type="range"
            min="1"
            max="10"
            value={miner.intensity}
            onChange={e => sendCommand(miner.minerId, { action: 'SetIntensity', level: parseInt(e.target.value) })}
            className="flex-1 accent-purple-500"
          />
          <span className="text-sm text-gray-500 w-6">10</span>
          <span className="text-lg font-bold text-purple-300 w-8 text-center">{miner.intensity}</span>
        </div>
      </div>

      {/* Command history */}
      {pendingCommands.length > 0 && (
        <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
          <h4 className="text-sm font-medium text-gray-400 mb-3">Recent Commands</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {[...pendingCommands].reverse().map(cmd => (
              <div key={cmd.commandId} className="flex items-center justify-between text-sm py-1">
                <span className="text-gray-300">{cmd.action.action}</span>
                <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                  cmd.status === 'success' ? 'bg-green-500/15 text-green-400' :
                  cmd.status === 'failed' ? 'bg-red-500/15 text-red-400' :
                  'bg-gray-500/15 text-gray-400'
                }`}>
                  {cmd.status === 'success' ? '✓' : cmd.status === 'failed' ? '✗' : '...'} {cmd.message || cmd.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Connection Tab ──────────────────────────────────────────────────────────

function ConnectionTab({ isConnected, miners, miningCommand }: {
  isConnected: boolean;
  miners: MinerInfo[];
  miningCommand: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(miningCommand).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="space-y-6">
      {/* Connection path */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-sm font-medium text-gray-400 mb-4">Connection Path</h4>
        <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
          <ConnectionNode label="Browser" icon={<Monitor className="w-4 h-4" />} active />
          <ChevronRight className="w-4 h-4 text-gray-600" />
          <ConnectionNode label="Tor" icon={<span className="text-sm">🧅</span>} active={isConnected} />
          <ChevronRight className="w-4 h-4 text-gray-600" />
          <ConnectionNode label="API Server" icon={<span className="text-sm">🌐</span>} active={isConnected} />
          <ChevronRight className="w-4 h-4 text-gray-600" />
          <ConnectionNode label="Miner" icon={<Pickaxe className="w-4 h-4" />} active={miners.length > 0} />
        </div>
      </div>

      {/* Connection status */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Status</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Relay</span>
            <p className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </p>
          </div>
          <div>
            <span className="text-gray-500">Miners Online</span>
            <p className="text-white">{miners.filter(m => m.isMining).length} / {miners.length}</p>
          </div>
          {miners.length > 0 && (
            <>
              <div>
                <span className="text-gray-500">Longest Uptime</span>
                <p className="text-white">{formatUptime(Math.max(...miners.map(m => m.uptimeSecs)))}</p>
              </div>
              <div>
                <span className="text-gray-500">Total Solutions</span>
                <p className="text-white">{miners.reduce((s, m) => s + m.solutions, 0)}</p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Setup instructions */}
      <div className="p-4 rounded-xl border bg-white/[0.02]" style={{ borderColor: 'rgba(139, 92, 246, 0.1)' }}>
        <h4 className="text-sm font-medium text-gray-400 mb-3">Setup Your Miner</h4>
        <p className="text-sm text-gray-400 mb-3">
          Download and run the miner binary. It will automatically connect to this wallet.
        </p>
        <div className="relative">
          <pre className="p-3 rounded-lg bg-black/30 text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap break-all">
            {miningCommand}
          </pre>
          <button
            onClick={handleCopy}
            className="absolute top-2 right-2 px-2 py-1 rounded text-xs bg-white/10 hover:bg-white/20 text-gray-300 transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Shared components ───────────────────────────────────────────────────────

function SummaryCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) {
  const colorMap: Record<string, string> = {
    purple: 'from-purple-500/10 to-purple-500/5 border-purple-500/15',
    green: 'from-green-500/10 to-green-500/5 border-green-500/15',
    amber: 'from-amber-500/10 to-amber-500/5 border-amber-500/15',
    blue: 'from-blue-500/10 to-blue-500/5 border-blue-500/15',
  };
  const textMap: Record<string, string> = {
    purple: 'text-purple-400',
    green: 'text-green-400',
    amber: 'text-amber-400',
    blue: 'text-blue-400',
  };

  return (
    <div className={`p-3 rounded-xl border bg-gradient-to-br ${colorMap[color] || colorMap.purple}`}>
      <div className={`mb-1 ${textMap[color] || textMap.purple}`}>{icon}</div>
      <div className="text-lg font-bold text-white">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function ConnectionNode({ label, icon, active }: { label: string; icon: React.ReactNode; active: boolean }) {
  return (
    <div className={`flex items-center gap-1.5 px-3 py-2 rounded-lg border ${
      active ? 'border-purple-500/30 bg-purple-500/10 text-purple-300' : 'border-gray-700 bg-white/5 text-gray-500'
    }`}>
      {icon}
      <span className="text-sm font-medium">{label}</span>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <Pickaxe className="w-12 h-12 text-gray-600 mb-4" />
      <p className="text-gray-400">{message}</p>
    </div>
  );
}

function formatLargeNumber(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

export default MinerLinkModal;
