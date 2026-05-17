import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Download, Upload, Pause, Play, Trash2, Plus, Link, FileText, RefreshCw, Settings, Zap, Users, Clock, HardDrive } from 'lucide-react';

const TORRENT_API = '/torrent-api';

interface TorrentInfo {
  info_hash: string;
  name: string;
  status: string;
  download_state: string;
  size: number;
  progress: number;
  download_rate: number;
  upload_rate: number;
  downloaded: number;
  uploaded: number;
  peers_connected: number;
  peers_total: number;
  seeds_connected: number;
  seeds_total: number;
  pieces_completed: number;
  pieces_total: number;
  eta: number | null;
  added_date: number;
  completed_date: number | null;
  download_dir: string;
}

function fmt_bytes(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB';
  return b + ' B';
}

function fmt_speed(bps: number): string {
  return fmt_bytes(bps) + '/s';
}

function fmt_eta(secs: number | null): string {
  if (!secs || secs <= 0) return '—';
  if (secs > 86400 * 7) return '>7d';
  if (secs > 86400) return Math.floor(secs / 86400) + 'd';
  if (secs > 3600) return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
  if (secs > 60) return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
  return secs + 's';
}

function state_color(state: string): string {
  switch (state) {
    case 'Downloading': return '#34D399';
    case 'Seeding': return '#60A5FA';
    case 'Completed': return '#A78BFA';
    case 'Paused': return '#FBBF24';
    case 'Error': return '#F87171';
    default: return '#94A3B8';
  }
}

async function api<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${TORRENT_API}${path}`, opts);
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// Magnet link dialog
function MagnetDialog({ onAdd, onClose }: { onAdd: (magnet: string) => void; onClose: () => void }) {
  const [value, setValue] = useState('');
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <motion.div
        className="relative rounded-2xl p-6 w-full max-w-lg"
        style={{ background: 'linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.98) 100%)', border: '1px solid rgba(212,175,55,0.3)' }}
        initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }}
      >
        <h3 className="text-lg font-bold text-amber-300 mb-4 flex items-center gap-2">
          <Link className="w-5 h-5" /> Add Magnet Link
        </h3>
        <textarea
          className="w-full rounded-xl p-3 text-sm font-mono text-slate-200 resize-none"
          style={{ background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(212,175,55,0.2)', minHeight: 80 }}
          placeholder="magnet:?xt=urn:btih:..."
          value={value}
          onChange={e => setValue(e.target.value)}
          autoFocus
        />
        <div className="flex gap-3 mt-4">
          <button onClick={onClose} className="flex-1 py-2 rounded-xl text-slate-400 border border-slate-600 hover:border-slate-400 transition-colors">
            Cancel
          </button>
          <button
            onClick={() => { if (value.trim()) { onAdd(value.trim()); onClose(); } }}
            disabled={!value.trim().startsWith('magnet:')}
            className="flex-1 py-2 rounded-xl font-semibold transition-all disabled:opacity-40"
            style={{ background: 'linear-gradient(135deg, #D4AF37, #FFD700)', color: '#0F172A' }}
          >
            Add Torrent
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

// Single torrent row
function TorrentRow({ t, onAction }: { t: TorrentInfo; onAction: (hash: string, action: string) => void }) {
  const isActive = t.download_state === 'Downloading';
  const isPaused = t.download_state === 'Paused';
  const isDone = t.download_state === 'Completed' || t.download_state === 'Seeding';

  return (
    <motion.div
      layout
      className="rounded-xl p-4"
      style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(212,175,55,0.1)' }}
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-200 truncate">{t.name || t.info_hash.slice(0, 16) + '…'}</p>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color: state_color(t.download_state), background: state_color(t.download_state) + '22' }}>
              {t.download_state}
            </span>
            <span className="text-xs text-slate-400">{fmt_bytes(t.size)}</span>
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <Users className="w-3 h-3" />{t.peers_connected}/{t.peers_total}
            </span>
            {t.eta && <span className="text-xs text-slate-400 flex items-center gap-1"><Clock className="w-3 h-3" />{fmt_eta(t.eta)}</span>}
          </div>
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          {isPaused && (
            <button onClick={() => onAction(t.info_hash, 'resume')} className="p-2 rounded-lg hover:bg-amber-500/20 text-amber-400 transition-colors" title="Resume">
              <Play className="w-4 h-4" />
            </button>
          )}
          {isActive && (
            <button onClick={() => onAction(t.info_hash, 'pause')} className="p-2 rounded-lg hover:bg-amber-500/20 text-amber-400 transition-colors" title="Pause">
              <Pause className="w-4 h-4" />
            </button>
          )}
          {!isActive && !isPaused && (
            <button onClick={() => onAction(t.info_hash, 'start')} className="p-2 rounded-lg hover:bg-green-500/20 text-green-400 transition-colors" title="Start">
              <Play className="w-4 h-4" />
            </button>
          )}
          <button onClick={() => { if (confirm(`Remove "${t.name}"?`)) onAction(t.info_hash, 'delete'); }} className="p-2 rounded-lg hover:bg-red-500/20 text-red-400 transition-colors" title="Delete">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 rounded-full overflow-hidden" style={{ background: 'rgba(30,41,59,0.8)' }}>
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: isDone ? 'linear-gradient(90deg, #6366F1, #A78BFA)' : 'linear-gradient(90deg, #D4AF37, #FFD700)' }}
          animate={{ width: `${(t.progress * 100).toFixed(1)}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>

      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-slate-400">{(t.progress * 100).toFixed(1)}% — {fmt_bytes(t.downloaded)} of {fmt_bytes(t.size)}</span>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          {t.download_rate > 0 && <span className="flex items-center gap-1 text-green-400"><Download className="w-3 h-3" />{fmt_speed(t.download_rate)}</span>}
          {t.upload_rate > 0 && <span className="flex items-center gap-1 text-blue-400"><Upload className="w-3 h-3" />{fmt_speed(t.upload_rate)}</span>}
        </div>
      </div>
    </motion.div>
  );
}

export default function TorrentTab() {
  const [torrents, setTorrents] = useState<TorrentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showMagnet, setShowMagnet] = useState(false);
  const [totalDown, setTotalDown] = useState(0);
  const [totalUp, setTotalUp] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const fetchTorrents = useCallback(async () => {
    try {
      const data = await api<TorrentInfo[]>('/torrents');
      setTorrents(data);
      setError(null);
      // Compute global rates
      let td = 0, tu = 0;
      for (const t of data) { td += t.download_rate; tu += t.upload_rate; }
      setTotalDown(td); setTotalUp(tu);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  // WebSocket for real-time updates
  useEffect(() => {
    fetchTorrents();

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/torrent-api/ws`;
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'TorrentUpdated' || msg.type === 'TorrentAdded') {
            setTorrents(prev => {
              const idx = prev.findIndex(t => t.info_hash === msg.data.info_hash);
              if (idx >= 0) { const n = [...prev]; n[idx] = msg.data; return n; }
              return [msg.data, ...prev];
            });
          } else if (msg.type === 'TorrentRemoved') {
            setTorrents(prev => prev.filter(t => t.info_hash !== msg.data));
          } else if (msg.type === 'StatsUpdate') {
            setTotalDown(msg.data.total_download_rate || 0);
            setTotalUp(msg.data.total_upload_rate || 0);
          }
        } catch {}
      };

      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 5000);
      };
    };

    connect();
    const poll = setInterval(fetchTorrents, 10000);

    return () => {
      ws?.close();
      clearTimeout(reconnectTimer);
      clearInterval(poll);
    };
  }, [fetchTorrents]);

  const handleAction = async (hash: string, action: string) => {
    try {
      if (action === 'delete') {
        await api(`/torrents/${hash}`, { method: 'DELETE' });
        setTorrents(prev => prev.filter(t => t.info_hash !== hash));
      } else {
        await api(`/torrents/${hash}/${action}`, { method: 'POST' });
        fetchTorrents();
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Action failed: ${msg}`);
    }
  };

  const handleAddMagnet = async (magnet: string) => {
    try {
      await api('/torrents/magnet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ magnet_uri: magnet, auto_start: true }),
      });
      fetchTorrents();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Failed to add magnet: ${msg}`);
    }
  };

  const handleFileUpload = async (file: File) => {
    const form = new FormData();
    form.append('torrent', file);
    try {
      await fetch(`${TORRENT_API}/torrents`, { method: 'POST', body: form }).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
      fetchTorrents();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Failed to add torrent file: ${msg}`);
    }
  };

  const active = torrents.filter(t => t.download_state === 'Downloading').length;
  const seeding = torrents.filter(t => t.download_state === 'Seeding').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-amber-300">Torrent Manager</h2>
          <p className="text-sm text-slate-400 mt-1">Software distribution & P2P downloads</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchTorrents}
            className="p-2 rounded-lg transition-colors hover:bg-amber-500/20 text-amber-400"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Global stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Active', value: active.toString(), icon: Zap, color: '#34D399' },
          { label: 'Seeding', value: seeding.toString(), icon: Upload, color: '#60A5FA' },
          { label: '↓ Speed', value: fmt_speed(totalDown), icon: Download, color: '#D4AF37' },
          { label: '↑ Speed', value: fmt_speed(totalUp), icon: Upload, color: '#A78BFA' },
        ].map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="rounded-xl p-4" style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(212,175,55,0.1)' }}>
            <div className="flex items-center gap-2 mb-1">
              <Icon className="w-4 h-4" style={{ color }} />
              <span className="text-xs text-slate-400">{label}</span>
            </div>
            <p className="text-xl font-bold" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Software distribution panel */}
      <div className="rounded-2xl p-5" style={{ background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(212,175,55,0.2)' }}>
        <h3 className="text-base font-semibold text-amber-300 mb-3 flex items-center gap-2">
          <HardDrive className="w-4 h-4" /> Software Distribution
        </h3>
        <p className="text-sm text-slate-400 mb-4">
          Seed the latest Q-NarwhalKnight binaries as torrents for redundant distribution.
          Users can download via magnet link from any seeding node.
        </p>
        <div className="space-y-2">
          {[
            { label: 'q-api-server (Linux x86_64)', path: '/downloads/q-api-server-linux-x86_64' },
            { label: 'q-miner (Linux x86_64)', path: '/downloads/q-miner-linux-x64' },
          ].map(({ label, path }) => (
            <div key={path} className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'rgba(30,41,59,0.6)' }}>
              <span className="text-sm text-slate-300 font-mono">{label}</span>
              <span className="text-xs text-slate-500 italic">Add .torrent file to seed</span>
            </div>
          ))}
        </div>
      </div>

      {/* Add torrent toolbar */}
      <div className="flex gap-3 flex-wrap">
        <motion.button
          onClick={() => fileRef.current?.click()}
          className="flex items-center gap-2 px-5 py-3 rounded-xl font-semibold text-sm transition-all"
          style={{ background: 'linear-gradient(135deg, rgba(212,175,55,0.2), rgba(255,215,0,0.15))', border: '1px solid rgba(212,175,55,0.4)', color: '#FFD700' }}
          whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
        >
          <FileText className="w-4 h-4" /> Add .torrent File
        </motion.button>
        <motion.button
          onClick={() => setShowMagnet(true)}
          className="flex items-center gap-2 px-5 py-3 rounded-xl font-semibold text-sm transition-all"
          style={{ background: 'rgba(96,165,250,0.15)', border: '1px solid rgba(96,165,250,0.4)', color: '#60A5FA' }}
          whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
        >
          <Link className="w-4 h-4" /> Add Magnet Link
        </motion.button>
        <input
          ref={fileRef}
          type="file"
          accept=".torrent"
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFileUpload(f); e.target.value = ''; }}
        />
      </div>

      {/* Torrent list */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div style={{ width: 36, height: 36, borderRadius: '50%', border: '3px solid rgba(212,175,55,0.2)', borderTopColor: '#d4af37', animation: 'spin 0.8s linear infinite' }} />
        </div>
      )}

      {error && !loading && (
        <div className="rounded-xl p-4 text-center" style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <p className="text-red-400 font-semibold mb-1">Cannot reach ZenTorrent backend</p>
          <p className="text-xs text-slate-400">{error}</p>
          <p className="text-xs text-slate-500 mt-2">Make sure the torrent backend is running on port 3040 and nginx proxies /torrent-api/</p>
        </div>
      )}

      {!loading && !error && torrents.length === 0 && (
        <div className="text-center py-12">
          <Download className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400 font-semibold">No torrents yet</p>
          <p className="text-sm text-slate-500 mt-1">Add a .torrent file or magnet link to get started</p>
        </div>
      )}

      {!loading && !error && torrents.length > 0 && (
        <div className="space-y-3">
          {torrents.map(t => (
            <TorrentRow key={t.info_hash} t={t} onAction={handleAction} />
          ))}
        </div>
      )}

      {/* Settings note */}
      {!loading && !error && (
        <div className="rounded-xl p-4 flex items-start gap-3" style={{ background: 'rgba(15,23,42,0.5)', border: '1px solid rgba(212,175,55,0.1)' }}>
          <Settings className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-semibold text-amber-300 mb-1">ZenTorrent settings</p>
            <p className="text-xs text-slate-400">Configure bandwidth limits, encryption, DHT/PEX, and download directory via the ZenTorrent API (<code className="text-slate-300">PUT /torrent-api/settings</code>). Default download directory: <code className="text-slate-300">/opt/orobit/shared/ZenTorrent/torrent-backend/downloads/</code></p>
          </div>
        </div>
      )}

      <AnimatePresence>
        {showMagnet && <MagnetDialog onAdd={handleAddMagnet} onClose={() => setShowMagnet(false)} />}
      </AnimatePresence>
    </div>
  );
}
