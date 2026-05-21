// Public q-flux edge-proxy stats pill for the global topbar.
//
// What you see: a small pill (always visible, no login required) with a
// pulse icon and one key number — req/s through the q-flux reverse
// proxy in front of every Quillon node.
//
// On hover: a beautiful dropdown with the full breakdown — req/s,
// active connections, healthy backends, uptime, TLS handshakes, error
// rate, HTTP/2 streams, bytes — each row with its own tooltip
// explaining what the metric is and why an operator would care.
//
// Data path: GET /api/v1/admin/flux/stats — the SAME endpoint the
// DeployControlPanel uses for its Analytics tab. The shape (FluxStats)
// matches that component exactly. For non-master-wallet (i.e. public)
// users this endpoint 401s; the component then shows '—' on the unknown
// fields with an EST badge so the reader knows the values are
// placeholders, not real telemetry.
//
// A small public-sanitised shim (req/s + active connections + backends-
// healthy only, no totals, no IPs) is the next obvious follow-up so the
// pill can show real numbers to anonymous visitors too. Tracked separately.

import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,    // pulse — main pill icon
  Radio,       // request rate
  Network,     // active connections
  Server,      // backend health
  Clock,       // uptime
  Lock,        // TLS handshakes
  AlertCircle, // error rate
  GitBranch,   // HTTP/2 streams
  Info,        // tooltip indicator
  CheckCircle2,
  XCircle,
} from 'lucide-react';

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

function formatUptime(seconds: number): string {
  if (!seconds || seconds < 0) return '—';
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function formatBytes(n: number): string {
  if (!n || n < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

function MetricRow({
  icon: Icon,
  label,
  value,
  estimated,
  tooltip,
  iconColor = 'text-emerald-400',
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  estimated: boolean;
  tooltip: string;
  iconColor?: string;
}) {
  return (
    <div
      className="flex items-center justify-between px-3 py-2 hover:bg-emerald-500/5 transition-colors group cursor-help"
      title={tooltip}
    >
      <div className="flex items-center gap-2.5">
        <Icon className={`w-3.5 h-3.5 ${iconColor}`} />
        <span className="text-slate-300 text-xs font-medium">{label}</span>
      </div>
      <div className="flex items-center gap-1.5">
        {estimated && (
          <span className="text-amber-400/70 text-[8px] uppercase tracking-wider font-bold px-1 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
            est
          </span>
        )}
        <span className="text-slate-100 text-xs font-mono font-bold">{value}</span>
        <Info className="w-3 h-3 text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
    </div>
  );
}

export function QFluxStatsPill() {
  const [stats, setStats] = useState<FluxStats | null>(null);
  const [estimated, setEstimated] = useState(true);
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Polling — every 5s. Uses the same endpoint as DeployControlPanel's
  // Analytics tab. For non-master sessions the endpoint 401s and we
  // gracefully fall through to estimated-fields display.
  useEffect(() => {
    let cancelled = false;

    // Matches the pattern from NodeSettingsModal/getAuthHeaders (v9.0.3) —
    // X-Wallet-Auth carries the address; Authorization: Bearer carries
    // either the OAuth token or the wallet address as fallback.
    function getAuthHeaders(): Record<string, string> {
      try {
        const wallet = localStorage.getItem('walletAddress') || '';
        const authToken = localStorage.getItem('authToken') || '';
        if (!wallet && !authToken) return { accept: 'application/json' };
        return {
          'X-Wallet-Auth': wallet,
          Authorization: `Bearer ${authToken || wallet}`,
          accept: 'application/json',
        };
      } catch { return { accept: 'application/json' }; }
    }

    async function tick() {
      try {
        const r = await fetch('/api/v1/admin/flux/stats', { headers: getAuthHeaders() });
        if (r.ok) {
          const j = await r.json();
          if (j?.data && !cancelled) {
            setStats(j.data as FluxStats);
            setEstimated(false);
            return;
          }
        }
      } catch { /* network or auth failure — fall through */ }
      if (!cancelled) setEstimated(true);
    }

    tick();
    const id = setInterval(tick, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  function handleEnter() {
    if (closeTimer.current) { clearTimeout(closeTimer.current); closeTimer.current = null; }
    setOpen(true);
  }
  function handleLeave() {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setOpen(false), 150);
  }

  const rps = stats?.requests_per_second ?? 0;
  const keyNumber = estimated || !stats
    ? '—'
    : rps >= 1000 ? `${(rps / 1000).toFixed(1)}k` : rps.toFixed(0);

  // Compose backend list from cluster info if present, else infer from
  // the known four-server topology.
  const backends: { name: string; healthy: boolean; rtt_ms?: number }[] = stats?.cluster
    ? [
        ...(stats.cluster.local_backends ?? []).map(b => ({
          name: b.addr.split(':')[0] || b.addr,
          healthy: b.healthy,
          rtt_ms: b.last_check_ms_ago,
        })),
        ...(stats.cluster.cluster_peers ?? []).map(b => ({
          name: b.addr.split(':')[0] || b.addr,
          healthy: b.healthy,
          rtt_ms: b.last_check_ms_ago,
        })),
      ]
    : [
        { name: 'Epsilon', healthy: !estimated },
        { name: 'Beta', healthy: !estimated },
        { name: 'Gamma', healthy: !estimated },
        { name: 'Delta', healthy: !estimated },
      ];

  const healthyBackends = backends.filter(b => b.healthy).length;
  const totalBackends = backends.length;

  return (
    <div
      className="relative"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      {/* The pill itself — matches the visual idiom of the Block / Peers / Network pills */}
      <motion.div
        className="flex flex-col items-center px-3 py-1 rounded-xl border min-w-[72px] cursor-pointer transition-colors"
        style={{
          background: 'rgba(16,185,129,0.08)',
          borderColor: open ? 'rgba(16,185,129,0.45)' : 'rgba(16,185,129,0.2)',
        }}
        whileHover={{ scale: 1.04 }}
        title="q-flux edge analytics — hover for full breakdown"
      >
        <span className="flex items-center gap-1 text-emerald-200 text-sm font-bold leading-tight font-mono">
          <Activity className="w-3 h-3 text-emerald-400" />
          {keyNumber}
        </span>
        <span className="text-emerald-400/50 text-[9px] font-semibold uppercase tracking-wider">req/s</span>
      </motion.div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.96 }}
            transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
            className="absolute top-full right-0 mt-2 w-80 rounded-2xl bg-slate-900/96 backdrop-blur-xl border border-emerald-500/25 shadow-2xl shadow-emerald-900/30 overflow-hidden z-50"
          >
            {/* Header */}
            <div className="px-4 py-3 border-b border-emerald-500/15 bg-gradient-to-r from-emerald-950/60 to-slate-900/40">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-100 text-sm font-bold">q-flux edge</span>
                  {stats?.version && !estimated && (
                    <span className="text-emerald-400/70 text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10">
                      v{stats.version}
                    </span>
                  )}
                </div>
                <span className="text-emerald-400/60 text-[10px] font-mono uppercase tracking-wider">
                  {estimated ? 'public' : 'live'}
                </span>
              </div>
              <p className="text-slate-400 text-[11px] mt-1 leading-snug">
                Custom Rust + io_uring reverse-proxy in front of every Quillon node.
              </p>
            </div>

            {/* Metric rows */}
            <div className="divide-y divide-slate-800/60">
              <MetricRow
                icon={Radio}
                label="Requests / sec"
                value={
                  estimated || !stats
                    ? '—'
                    : stats.requests_per_second >= 1000
                      ? `${(stats.requests_per_second / 1000).toFixed(2)}k`
                      : stats.requests_per_second.toFixed(0)
                }
                estimated={estimated}
                tooltip="HTTP requests per second across all backends, counted at the q-flux ingress layer before any caching. Refreshed every 5s."
                iconColor="text-emerald-400"
              />
              <MetricRow
                icon={Network}
                label="Active connections"
                value={
                  estimated || !stats
                    ? '—'
                    : stats.active_connections.toLocaleString()
                }
                estimated={estimated}
                tooltip="Open TCP connections currently held by q-flux. Includes keep-alive idle and HTTP/2 multiplexed connections in the pool."
                iconColor="text-cyan-400"
              />
              <MetricRow
                icon={Server}
                label="Backends healthy"
                value={`${healthyBackends} / ${totalBackends}`}
                estimated={estimated}
                tooltip="Upstream q-api-server nodes passing health checks. Quillon's HA topology is Beta + Gamma + Delta + Epsilon."
                iconColor={healthyBackends === totalBackends ? 'text-emerald-400' : 'text-amber-400'}
              />
              <MetricRow
                icon={Clock}
                label="Uptime"
                value={stats && !estimated ? formatUptime(stats.uptime_secs) : '—'}
                estimated={estimated}
                tooltip="How long the active q-flux process has been running without restart. Frequent restarts indicate deploys or crashes."
                iconColor="text-violet-400"
              />
              <MetricRow
                icon={AlertCircle}
                label="Error rate"
                value={
                  estimated || !stats
                    ? '—'
                    : `${stats.error_rate_pct.toFixed(2)}%`
                }
                estimated={estimated}
                tooltip="Fraction of HTTP responses with 4xx or 5xx status. <1% = healthy; >5% = investigate (rate-limiting, backend failures, malformed clients)."
                iconColor={
                  !estimated && stats && stats.error_rate_pct > 5
                    ? 'text-rose-400'
                    : 'text-blue-400'
                }
              />
              <MetricRow
                icon={Lock}
                label="TLS handshakes"
                value={
                  estimated || !stats
                    ? '—'
                    : stats.tls_handshakes.toLocaleString()
                }
                estimated={estimated}
                tooltip="Total TLS handshakes since q-flux started. High growth rate means clients aren't reusing keep-alive connections."
                iconColor="text-rose-400"
              />
              <MetricRow
                icon={GitBranch}
                label="HTTP/2 streams"
                value={
                  estimated || !stats
                    ? '—'
                    : stats.h2_streams_opened.toLocaleString()
                }
                estimated={estimated}
                tooltip="HTTP/2 streams opened since startup. HTTP/2 multiplexing is how q-flux serves thousands of concurrent miners on a small connection pool."
                iconColor="text-indigo-400"
              />
              <MetricRow
                icon={Activity}
                label="Bytes / total"
                value={
                  estimated || !stats
                    ? '—'
                    : `${formatBytes(stats.bytes_received)} ↑ ${formatBytes(stats.bytes_sent)} ↓`
                }
                estimated={estimated}
                tooltip="Total bytes received from clients (uploads — mining shares, tx submits) and sent back (downloads — SSE streams, status responses)."
                iconColor="text-amber-400"
              />
            </div>

            {/* Backends list */}
            {backends.length > 0 && (
              <div className="border-t border-slate-800/60 px-3 py-2.5 bg-slate-950/40">
                <p className="text-slate-500 text-[10px] uppercase tracking-wider font-bold mb-1.5">
                  Backends
                </p>
                <div className="grid grid-cols-2 gap-1.5">
                  {backends.map((b) => (
                    <div
                      key={b.name + (b.rtt_ms ?? '')}
                      className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-900/60"
                      title={`${b.name}${b.healthy ? ' — healthy' : ' — unhealthy'}${b.rtt_ms !== undefined && b.rtt_ms > 0 ? ` · last check ${b.rtt_ms}ms ago` : ''}`}
                    >
                      {b.healthy
                        ? <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        : <XCircle className="w-3 h-3 text-rose-400" />}
                      <span className="text-slate-300 text-[10px] font-mono">{b.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Footer */}
            <div className="px-4 py-2.5 border-t border-slate-800/60 bg-slate-950/50">
              <p className="text-slate-500 text-[10px] leading-relaxed">
                {estimated
                  ? 'Sign in with the master wallet to see live telemetry. Public visitors see this placeholder until the public shim endpoint lands.'
                  : `Polled every 5s from /api/v1/admin/flux/stats · workers: ${stats?.worker_count ?? '?'}`}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default QFluxStatsPill;
