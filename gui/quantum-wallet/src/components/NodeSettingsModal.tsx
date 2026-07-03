// v9.0.3: Node Settings Modal — Admin modal with OAuth2-aware auth + slider-based parameter adjustment
// Listens for 'open-node-settings' custom event from TopBar gear icon
// Works after OAuth2 login by checking both wallet auth AND OAuth2 access tokens

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Settings, Shield, Globe, Key, Trash2, RefreshCw, Clock, Server, Wifi, DollarSign, Download, ArrowUpCircle, CheckCircle, AlertTriangle, Loader2, Sliders, Zap, Activity } from 'lucide-react';

const MASTER_WALLET = 'efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723';

interface AdminSettings {
  admin_wallet: string;
  version: string;
  uptime_secs: number;
  height: number;
  network_height: number;
  peers: number;
  network_id: string;
  oauth2_clients: number;
  oauth2_active_tokens: number;
  oauth2_consents: number;
}

interface ConsentEntry {
  client_id: string;
  scopes: string[];
  granted_at: string;
}

interface NodeInfo {
  version: string;
  uptime_secs: number;
  height: number;
  network_height: number;
  peers: number;
  network_id: string;
  mining_healthy: boolean;
}

interface OperatorFees {
  node_operator_fee_promille: number;
  node_operator_fee_percent: string;
  dex_protocol_fee_bps: number;
  dex_protocol_fee_percent: string;
  admin_wallet: string;
  admin_wallet_balance_qug: number;
  founder_wallet_balance_qug: number;
}

interface FeeEarnings {
  admin_wallet: string;
  fee_share_promille: number;
  fee_share_percent: string;
  session_earnings_qug: number;
  total_earnings_qug: number;
  fee_tx_count: number;
  node_uptime_secs: number;
}

interface NodeUpdateInfo {
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  download_url: string | null;
}

interface AutoUpdateStatus {
  auto_update_enabled: boolean;
  current_version: string;
  state: NodeUpdateStateData;
  notification_email: string | null;
}

type NodeUpdateStateData =
  | { state: 'Disabled' }
  | { state: 'Idle' }
  | { state: 'WaitingForQuorum'; version: string; signers_so_far: number; signers_needed: number }
  | { state: 'Available'; version: string; download_url: string }
  | { state: 'Downloading'; version: string; progress_percent: number }
  | { state: 'Verifying'; version: string }
  | { state: 'PreflightCheck'; version: string }
  | { state: 'ReadyToRestart'; version: string }
  | { state: 'RestartScheduled'; version: string; restart_in_secs: number }
  | { state: 'Error'; version: string; message: string; retry_count: number }
  | { state: 'RollingBack'; version: string; reason: string };

type TabId = 'overview' | 'parameters' | 'oauth2' | 'node' | 'updates';

function formatUptime(secs: number): string {
  const days = Math.floor(secs / 86400);
  const hours = Math.floor((secs % 86400) / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

// v9.0.3: Smart auth headers — uses OAuth2 token when available, falls back to wallet address
function getAuthHeaders(): Record<string, string> {
  const wallet = localStorage.getItem('walletAddress') || '';
  const authToken = localStorage.getItem('authToken') || '';
  const headers: Record<string, string> = {
    'X-Wallet-Auth': wallet,
    'Content-Type': 'application/json',
  };
  // Prefer OAuth2 access token over raw wallet address for Bearer
  if (authToken && authToken.length > 0) {
    headers['Authorization'] = `Bearer ${authToken}`;
  } else {
    headers['Authorization'] = `Bearer ${wallet}`;
  }
  return headers;
}

function checkIsMasterWallet(): boolean {
  const wallet = localStorage.getItem('walletAddress') || '';
  const clean = wallet.replace('qnk', '').replace('qug', '');
  return clean === MASTER_WALLET;
}

export default function NodeSettingsModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [consents, setConsents] = useState<ConsentEntry[]>([]);
  const [nodeInfo, setNodeInfo] = useState<NodeInfo | null>(null);
  const [operatorFees, setOperatorFees] = useState<OperatorFees | null>(null);
  const [updateInfo, setUpdateInfo] = useState<NodeUpdateInfo | null>(null);
  const [autoUpdateStatus, setAutoUpdateStatus] = useState<AutoUpdateStatus | null>(null);
  const [feeEarnings, setFeeEarnings] = useState<FeeEarnings | null>(null);
  const [loading, setLoading] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const isMaster = checkIsMasterWallet();

  // Listen for open event
  useEffect(() => {
    const handler = () => setIsOpen(true);
    window.addEventListener('open-node-settings', handler);
    return () => window.removeEventListener('open-node-settings', handler);
  }, []);

  const fetchSettings = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/settings', { headers: getAuthHeaders() });
      if (res.status === 403) {
        setError('Not authorized. Start your node with --admin-wallet YOUR_WALLET to enable admin access.');
        return;
      }
      if (res.ok) setSettings(await res.json());
    } catch {
      setError('Could not connect to node API');
    }
  }, []);

  const fetchConsents = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/oauth2/consents', { headers: getAuthHeaders() });
      if (res.ok) setConsents(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  const fetchNodeInfo = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/node/info', { headers: getAuthHeaders() });
      if (res.ok) setNodeInfo(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  const fetchOperatorFees = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/operator-fees', { headers: getAuthHeaders() });
      if (res.ok) setOperatorFees(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  const fetchUpdateInfo = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/node/update-check', { headers: getAuthHeaders() });
      if (res.ok) setUpdateInfo(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  const fetchAutoUpdateStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/update/status');
      if (res.ok) setAutoUpdateStatus(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  const fetchFeeEarnings = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/admin/fee-earnings', { headers: getAuthHeaders() });
      if (res.ok) setFeeEarnings(await res.json());
    } catch { /* ignore - non-critical */ }
  }, []);

  // Fetch data when modal opens
  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    Promise.all([fetchSettings(), fetchConsents(), fetchNodeInfo(), fetchOperatorFees(), fetchUpdateInfo(), fetchAutoUpdateStatus(), fetchFeeEarnings()])
      .finally(() => setLoading(false));
  }, [isOpen, fetchSettings, fetchConsents, fetchNodeInfo, fetchOperatorFees, fetchUpdateInfo, fetchAutoUpdateStatus, fetchFeeEarnings]);

  // Auto-refresh update status when Updates tab is active
  useEffect(() => {
    if (!isOpen || activeTab !== 'updates') return;
    const interval = setInterval(fetchAutoUpdateStatus, 5000);
    return () => clearInterval(interval);
  }, [isOpen, activeTab, fetchAutoUpdateStatus]);

  const handleRevoke = async (clientId: string) => {
    setRevoking(clientId);
    try {
      const res = await fetch('/api/v1/admin/oauth2/revoke-consent', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ client_id: clientId }),
      });
      if (res.ok) {
        setConsents(prev => prev.filter(c => c.client_id !== clientId));
      }
    } catch { /* ignore */ }
    setRevoking(null);
  };

  const handleRefresh = () => {
    setLoading(true);
    setSaveStatus(null);
    Promise.all([fetchSettings(), fetchConsents(), fetchNodeInfo(), fetchOperatorFees(), fetchUpdateInfo(), fetchAutoUpdateStatus(), fetchFeeEarnings()])
      .finally(() => setLoading(false));
  };

  if (!isOpen) return null;

  const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <Settings className="w-4 h-4" /> },
    { id: 'parameters', label: 'Parameters', icon: <Sliders className="w-4 h-4" /> },
    { id: 'oauth2', label: 'OAuth2', icon: <Key className="w-4 h-4" /> },
    { id: 'node', label: 'Node', icon: <Server className="w-4 h-4" /> },
    { id: 'updates', label: 'Updates', icon: <ArrowUpCircle className="w-4 h-4" /> },
  ];

  const syncPct = settings && settings.network_height > 0
    ? Math.min(100, (settings.height / settings.network_height) * 100)
    : 100;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-[9999] flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal */}
          <motion.div
            className="relative w-full max-w-5xl mx-auto bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-blue-500/30 rounded-2xl shadow-2xl overflow-hidden"
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-blue-500/20">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-500/20 rounded-lg">
                  <Settings className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Node Settings</h2>
                  <p className="text-xs text-slate-400">
                    {isMaster ? 'Master admin' : 'Node operator'} configuration
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {saveStatus && (
                  <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full animate-pulse ${
                    saveStatus === 'saved' ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : saveStatus === 'error' ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                    : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  }`}>
                    {saveStatus === 'saved' ? 'SAVED' : saveStatus === 'error' ? 'ERROR' : 'SAVING...'}
                  </span>
                )}
                {updateInfo?.update_available && (
                  <span className="px-2 py-0.5 text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full animate-pulse">
                    UPDATE
                  </span>
                )}
                <button
                  onClick={handleRefresh}
                  className="p-2 rounded-lg hover:bg-slate-700/50 transition-colors"
                  title="Refresh"
                >
                  <RefreshCw className={`w-4 h-4 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 rounded-lg hover:bg-slate-700/50 transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-slate-700/50 overflow-x-auto">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/5'
                      : 'text-slate-400 hover:text-slate-300 hover:bg-slate-700/20'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="p-6 max-h-[80vh] overflow-y-auto">
              {error ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-4">
                    <Settings className="w-7 h-7 text-amber-400/60" />
                  </div>
                  <p className="text-slate-300 font-medium mb-2">Admin Access Required</p>
                  <p className="text-sm text-slate-500 max-w-sm">{error}</p>
                  <code className="mt-4 px-3 py-2 bg-slate-800/80 border border-slate-700/50 rounded-lg text-xs text-slate-400 font-mono">
                    ./q-api-server --admin-wallet YOUR_WALLET
                  </code>
                </div>
              ) : loading && !settings ? (
                <div className="flex items-center justify-center py-12">
                  <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
                  <span className="ml-3 text-slate-400">Loading...</span>
                </div>
              ) : activeTab === 'overview' ? (
                <OverviewTab settings={settings} syncPct={syncPct} feeEarnings={feeEarnings} />
              ) : activeTab === 'parameters' ? (
                <ParametersTab
                  fees={operatorFees}
                  isMaster={isMaster}
                  onSaveStatus={setSaveStatus}
                  onRefreshFees={fetchOperatorFees}
                />
              ) : activeTab === 'oauth2' ? (
                <OAuth2Tab
                  settings={settings}
                  consents={consents}
                  revoking={revoking}
                  onRevoke={handleRevoke}
                />
              ) : activeTab === 'updates' ? (
                <UpdatesTab autoUpdateStatus={autoUpdateStatus} updateInfo={updateInfo} onToggle={async () => {
                  try {
                    const res = await fetch('/api/v1/admin/update/toggle', {
                      method: 'POST',
                      headers: getAuthHeaders(),
                    });
                    if (res.ok) {
                      const data = await res.json();
                      setAutoUpdateStatus(prev => prev ? { ...prev, auto_update_enabled: data.auto_update_enabled } : prev);
                    }
                  } catch { /* ignore */ }
                }} onSetEmail={async (email: string | null) => {
                  try {
                    const res = await fetch('/api/v1/admin/update/notification-email', {
                      method: 'POST',
                      headers: getAuthHeaders(),
                      body: JSON.stringify({ email }),
                    });
                    if (res.ok) {
                      const data = await res.json();
                      setAutoUpdateStatus(prev => prev ? { ...prev, notification_email: data.notification_email } : prev);
                    }
                  } catch { /* ignore */ }
                }} />
              ) : (
                <NodeTab nodeInfo={nodeInfo} updateInfo={updateInfo} onCheckUpdate={fetchUpdateInfo} />
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}

// -- Reusable Components --

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub?: string }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-xl font-bold text-white truncate">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1 truncate">{sub}</div>}
    </div>
  );
}

// v9.0.3: Slider component with label, value display, and color coding
function ParamSlider({
  label,
  description,
  value,
  min,
  max,
  step,
  displayValue,
  displaySuffix,
  color,
  disabled,
  onChange,
}: {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  displaySuffix?: string;
  color: 'blue' | 'amber' | 'cyan' | 'emerald' | 'purple';
  disabled?: boolean;
  onChange: (val: number) => void;
}) {
  const colorMap = {
    blue: { track: 'from-blue-500 to-blue-400', text: 'text-blue-400', bg: 'bg-blue-500/20', border: 'border-blue-500/30' },
    amber: { track: 'from-amber-500 to-amber-400', text: 'text-amber-400', bg: 'bg-amber-500/20', border: 'border-amber-500/30' },
    cyan: { track: 'from-cyan-500 to-cyan-400', text: 'text-cyan-400', bg: 'bg-cyan-500/20', border: 'border-cyan-500/30' },
    emerald: { track: 'from-emerald-500 to-emerald-400', text: 'text-emerald-400', bg: 'bg-emerald-500/20', border: 'border-emerald-500/30' },
    purple: { track: 'from-purple-500 to-purple-400', text: 'text-purple-400', bg: 'bg-purple-500/20', border: 'border-purple-500/30' },
  };
  const c = colorMap[color];
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;

  return (
    <div className={`${c.bg} border ${c.border} rounded-xl p-4 ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-slate-200">{label}</span>
        <span className={`text-lg font-bold font-mono ${c.text}`}>
          {displayValue}{displaySuffix && <span className="text-xs text-slate-400 ml-1">{displaySuffix}</span>}
        </span>
      </div>
      <p className="text-xs text-slate-500 mb-3">{description}</p>
      <div className="relative">
        <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${c.track} rounded-full transition-all duration-150`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={e => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-2 opacity-0 cursor-pointer disabled:cursor-not-allowed"
          style={{ top: '0px' }}
        />
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-slate-600">{min}</span>
        <span className="text-[10px] text-slate-600">{max}</span>
      </div>
    </div>
  );
}

// -- Overview Tab --

function OverviewTab({ settings, syncPct, feeEarnings }: { settings: AdminSettings | null; syncPct: number; feeEarnings: FeeEarnings | null }) {
  if (!settings) return <p className="text-slate-400">No data available</p>;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Shield className="w-4 h-4 text-emerald-400" />}
          label="Admin Wallet"
          value={`${settings.admin_wallet.slice(0, 8)}...${settings.admin_wallet.slice(-8)}`}
          sub={settings.admin_wallet}
        />
        <StatCard
          icon={<Globe className="w-4 h-4 text-blue-400" />}
          label="Network"
          value={settings.network_id}
          sub={`v${settings.version}`}
        />
        <StatCard
          icon={<Clock className="w-4 h-4 text-amber-400" />}
          label="Uptime"
          value={formatUptime(settings.uptime_secs)}
        />
        <StatCard
          icon={<Wifi className="w-4 h-4 text-green-400" />}
          label="Peers"
          value={String(settings.peers)}
        />
      </div>

      {/* Fee Earnings */}
      <div className="bg-gradient-to-br from-emerald-900/30 to-slate-800/50 border border-emerald-700/40 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-emerald-300 mb-3 flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-emerald-400" /> Fee Earnings
        </h3>
        {feeEarnings ? (
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center">
              <div className="text-lg font-bold text-emerald-300">{feeEarnings.total_earnings_qug.toFixed(4)}</div>
              <div className="text-xs text-slate-500">Total Earned (QUG)</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">{feeEarnings.session_earnings_qug.toFixed(4)}</div>
              <div className="text-xs text-slate-500">This Session</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-white">{feeEarnings.fee_tx_count}</div>
              <div className="text-xs text-slate-500">Fee Transactions</div>
            </div>
          </div>
        ) : (
          <div className="text-center py-2">
            <div className="text-sm text-slate-400">No earnings data yet</div>
            <div className="text-xs text-slate-500 mt-1">Fees are earned when users trade on the DEX through your node</div>
          </div>
        )}
        {feeEarnings && (
          <div className="mt-3 pt-3 border-t border-slate-700/50 flex items-center justify-between">
            <span className="text-xs text-slate-500">Your fee share: {feeEarnings.fee_share_percent}</span>
            <span className="text-xs text-slate-500">Uptime: {formatUptime(feeEarnings.node_uptime_secs)}</span>
          </div>
        )}
      </div>

      {/* Sync progress */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Sync Progress</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            syncPct >= 99.5 ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'
          }`}>
            {syncPct >= 99.5 ? 'Synced' : `${syncPct.toFixed(1)}%`}
          </span>
        </div>
        <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full transition-all"
            style={{ width: `${syncPct}%` }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-xs text-slate-500">Local: {settings.height.toLocaleString()}</span>
          <span className="text-xs text-slate-500">Network: {settings.network_height.toLocaleString()}</span>
        </div>
      </div>

      {/* OAuth2 summary */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Key className="w-4 h-4 text-purple-400" /> OAuth2 Summary
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center">
            <div className="text-lg font-bold text-white">{settings.oauth2_clients}</div>
            <div className="text-xs text-slate-500">Clients</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-white">{settings.oauth2_active_tokens}</div>
            <div className="text-xs text-slate-500">Active Tokens</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-white">{settings.oauth2_consents}</div>
            <div className="text-xs text-slate-500">Consents</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// -- Parameters Tab (v9.0.3: Slider-based parameter adjustment) --

function ParametersTab({
  fees,
  isMaster,
  onSaveStatus,
  onRefreshFees,
}: {
  fees: OperatorFees | null;
  isMaster: boolean;
  onSaveStatus: (status: string | null) => void;
  onRefreshFees: () => Promise<void>;
}) {
  const [operatorFee, setOperatorFee] = useState(0); // promille 0-500
  const [dexFee, setDexFee] = useState(0); // bps 0-10
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync slider values from fetched data
  useEffect(() => {
    if (fees) {
      setOperatorFee(fees.node_operator_fee_promille);
      setDexFee(fees.dex_protocol_fee_bps);
      setDirty(false);
    }
  }, [fees]);

  const handleOperatorFeeChange = (val: number) => {
    setOperatorFee(val);
    setDirty(true);
  };

  const handleDexFeeChange = (val: number) => {
    setDexFee(val);
    setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    onSaveStatus('saving');
    try {
      const body: Record<string, number> = {};
      if (fees && operatorFee !== fees.node_operator_fee_promille) body.node_operator_fee_promille = operatorFee;
      if (fees && dexFee !== fees.dex_protocol_fee_bps) body.dex_protocol_fee_bps = dexFee;

      if (Object.keys(body).length === 0) {
        onSaveStatus('saved');
        setSaving(false);
        setDirty(false);
        return;
      }

      const res = await fetch('/api/v1/admin/operator-fees', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(body),
      });
      if (res.ok) {
        onSaveStatus('saved');
        setDirty(false);
        await onRefreshFees();
      } else {
        const text = await res.text().catch(() => 'Unknown error');
        onSaveStatus('error');
        console.error('Failed to save fees:', res.status, text);
      }
    } catch (err) {
      onSaveStatus('error');
      console.error('Save error:', err);
    }
    setSaving(false);
    // Clear status after 3s
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => onSaveStatus(null), 3000);
  };

  const handleReset = () => {
    if (fees) {
      setOperatorFee(fees.node_operator_fee_promille);
      setDexFee(fees.dex_protocol_fee_bps);
      setDirty(false);
      onSaveStatus(null);
    }
  };

  if (!fees) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Sliders className="w-10 h-10 text-slate-600 mb-4" />
        <p className="text-slate-300 font-medium mb-2">Fee parameters not available</p>
        <p className="text-sm text-slate-500">Requires master wallet or node admin access</p>
      </div>
    );
  }

  const operatorFeePct = (operatorFee / 10).toFixed(1);
  const dexFeePct = (dexFee / 100).toFixed(2);
  const treasuryPct = (100 - operatorFee / 10).toFixed(1);

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-2">
        <Sliders className="w-5 h-5 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-200">Fee & Revenue Parameters</h3>
        {!isMaster && (
          <span className="ml-auto px-2 py-0.5 text-[10px] font-medium bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full">
            READ ONLY
          </span>
        )}
      </div>

      {/* Wallet balances */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Shield className="w-4 h-4 text-emerald-400" />}
          label="Founder Balance"
          value={`${fees.founder_wallet_balance_qug.toFixed(4)} QUG`}
        />
        <StatCard
          icon={<DollarSign className="w-4 h-4 text-amber-400" />}
          label="Operator Balance"
          value={`${fees.admin_wallet_balance_qug.toFixed(4)} QUG`}
          sub={`${fees.admin_wallet.slice(0, 8)}...${fees.admin_wallet.slice(-6)}`}
        />
      </div>

      {/* Operator Fee Slider */}
      <ParamSlider
        label="Node Operator Fee"
        description={`Your share of transaction fees. Treasury gets ${treasuryPct}%.`}
        value={operatorFee}
        min={0}
        max={500}
        step={10}
        displayValue={`${operatorFeePct}%`}
        color="amber"
        disabled={!isMaster}
        onChange={handleOperatorFeeChange}
      />

      {/* DEX Protocol Fee Slider */}
      <ParamSlider
        label="DEX Protocol Fee"
        description="Extracted from each DEX swap. Split between treasury and operator based on fee share above."
        value={dexFee}
        min={0}
        max={10}
        step={1}
        displayValue={`${dexFeePct}%`}
        displaySuffix={`(${dexFee} bps)`}
        color="cyan"
        disabled={!isMaster}
        onChange={handleDexFeeChange}
      />

      {/* Fee Distribution Preview */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3 flex items-center gap-2">
          <Activity className="w-3.5 h-3.5" /> Fee Distribution Preview
        </h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Treasury Share</span>
            <span className="text-xs font-mono text-white">{treasuryPct}%</span>
          </div>
          <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden flex">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-150"
              style={{ width: `${100 - operatorFee / 5}%` }}
            />
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-amber-400 transition-all duration-150"
              style={{ width: `${operatorFee / 5}%` }}
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-[10px] text-slate-500">Treasury ({treasuryPct}%)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-500">Operator ({operatorFeePct}%)</span>
              <div className="w-2 h-2 rounded-full bg-amber-400" />
            </div>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-700/30 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">DEX Protocol Fee</span>
            <span className="text-slate-400">{dexFeePct}% per swap</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-500">LP Fee (fixed)</span>
            <span className="text-slate-400">0.30% (stays in pool)</span>
          </div>
        </div>
      </div>

      {/* Save / Reset buttons */}
      {isMaster && dirty && (
        <div className="flex gap-3">
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 transition-all"
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
            Save Changes
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-3 text-sm font-medium text-slate-400 bg-slate-700/50 border border-slate-600/50 rounded-xl hover:bg-slate-600/50 transition-colors"
          >
            Reset
          </button>
        </div>
      )}

      {!isMaster && (
        <div className="bg-slate-800/30 border border-slate-700/30 rounded-lg p-3 text-center">
          <p className="text-xs text-slate-500">
            Only the master wallet can modify parameters.
            You are viewing current settings as a node operator.
          </p>
        </div>
      )}
    </div>
  );
}

// -- OAuth2 Tab --

function OAuth2Tab({
  settings,
  consents,
  revoking,
  onRevoke,
}: {
  settings: AdminSettings | null;
  consents: ConsentEntry[];
  revoking: string | null;
  onRevoke: (clientId: string) => void;
}) {
  return (
    <div className="space-y-4">
      {consents.length === 0 ? (
        <div className="text-center py-8">
          <Key className="w-8 h-8 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No OAuth2 consents granted</p>
          <p className="text-xs text-slate-500 mt-1">Third-party apps you authorize will appear here</p>
        </div>
      ) : (
        consents.map(consent => (
          <div
            key={consent.client_id}
            className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 flex items-center justify-between"
          >
            <div>
              <div className="text-sm font-medium text-white">{consent.client_id}</div>
              <div className="text-xs text-slate-400 mt-1">
                Scopes: {consent.scopes.join(', ') || 'none'}
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                Granted: {new Date(consent.granted_at).toLocaleDateString()}
              </div>
            </div>
            <button
              onClick={() => onRevoke(consent.client_id)}
              disabled={revoking === consent.client_id}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg hover:bg-red-500/20 disabled:opacity-50 transition-colors"
            >
              {revoking === consent.client_id ? (
                <RefreshCw className="w-3 h-3 animate-spin" />
              ) : (
                <Trash2 className="w-3 h-3" />
              )}
              Revoke
            </button>
          </div>
        ))
      )}

      {settings && (
        <div className="bg-slate-800/30 border border-slate-700/30 rounded-lg p-3 text-xs text-slate-500">
          {settings.oauth2_clients} registered client{settings.oauth2_clients !== 1 ? 's' : ''},{' '}
          {settings.oauth2_active_tokens} active token{settings.oauth2_active_tokens !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

// -- Node Tab (with update check) --

function NodeTab({ nodeInfo, updateInfo, onCheckUpdate }: {
  nodeInfo: NodeInfo | null;
  updateInfo: NodeUpdateInfo | null;
  onCheckUpdate: () => void;
}) {
  const [checking, setChecking] = useState(false);

  if (!nodeInfo) return <p className="text-slate-400">No data available</p>;

  const syncPct = nodeInfo.network_height > 0
    ? Math.min(100, (nodeInfo.height / nodeInfo.network_height) * 100)
    : 100;

  const handleCheckUpdate = async () => {
    setChecking(true);
    await onCheckUpdate();
    setChecking(false);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Server className="w-4 h-4 text-blue-400" />}
          label="Version"
          value={`v${nodeInfo.version}`}
        />
        <StatCard
          icon={<Clock className="w-4 h-4 text-amber-400" />}
          label="Uptime"
          value={formatUptime(nodeInfo.uptime_secs)}
        />
        <StatCard
          icon={<Wifi className="w-4 h-4 text-green-400" />}
          label="Peers"
          value={String(nodeInfo.peers)}
        />
        <StatCard
          icon={<Globe className="w-4 h-4 text-cyan-400" />}
          label="Network"
          value={nodeInfo.network_id}
        />
      </div>

      {/* Block height */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Block Height</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            syncPct >= 99.5
              ? 'bg-green-500/20 text-green-400'
              : 'bg-amber-500/20 text-amber-400'
          }`}>
            {syncPct >= 99.5 ? 'Synced' : `${syncPct.toFixed(1)}%`}
          </span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-white font-mono">{nodeInfo.height.toLocaleString()}</span>
          <span className="text-sm text-slate-500">/ {nodeInfo.network_height.toLocaleString()}</span>
        </div>
      </div>

      {/* Mining status */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 flex items-center gap-3">
        <div className={`w-3 h-3 rounded-full ${nodeInfo.mining_healthy ? 'bg-green-400' : 'bg-red-400'}`} />
        <div>
          <div className="text-sm font-medium text-white">
            Mining {nodeInfo.mining_healthy ? 'Healthy' : 'Degraded'}
          </div>
          <div className="text-xs text-slate-500">
            Block production is {nodeInfo.mining_healthy ? 'operating normally' : 'experiencing issues'}
          </div>
        </div>
      </div>

      {/* Node Update Section */}
      <div className={`bg-slate-800/50 border rounded-xl p-4 ${
        updateInfo?.update_available
          ? 'border-amber-500/40 bg-amber-500/5'
          : 'border-slate-700/50'
      }`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <ArrowUpCircle className="w-4 h-4 text-cyan-400" /> Software Update
          </h3>
          <button
            onClick={handleCheckUpdate}
            disabled={checking}
            className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium text-slate-300 bg-slate-700/50 border border-slate-600/50 rounded-lg hover:bg-slate-600/50 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`w-3 h-3 ${checking ? 'animate-spin' : ''}`} />
            Check
          </button>
        </div>

        {updateInfo ? (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Current</span>
              <span className="text-xs font-mono text-white">v{updateInfo.current_version}</span>
            </div>
            {updateInfo.latest_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Latest</span>
                <span className={`text-xs font-mono ${updateInfo.update_available ? 'text-amber-400' : 'text-green-400'}`}>
                  v{updateInfo.latest_version}
                </span>
              </div>
            )}
            {updateInfo.update_available ? (
              <a
                href={updateInfo.download_url || '#'}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 flex items-center justify-center gap-2 w-full px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-amber-600 to-orange-600 rounded-lg hover:from-amber-500 hover:to-orange-500 transition-all"
              >
                <Download className="w-4 h-4" />
                Download v{updateInfo.latest_version}
              </a>
            ) : (
              <div className="mt-2 flex items-center gap-2 text-xs text-green-400">
                <CheckCircle className="w-3.5 h-3.5" />
                Node is up to date
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-slate-500">Click &quot;Check&quot; to see if an update is available</p>
        )}
      </div>
    </div>
  );
}

// -- Updates Tab --

function getStateLabel(state: string): string {
  const labels: Record<string, string> = {
    Disabled: 'Disabled',
    Idle: 'Idle',
    WaitingForQuorum: 'Waiting for Quorum',
    Available: 'Update Available',
    Downloading: 'Downloading',
    Verifying: 'Verifying',
    PreflightCheck: 'Preflight Check',
    ReadyToRestart: 'Ready to Restart',
    RestartScheduled: 'Restart Scheduled',
    Error: 'Error',
    RollingBack: 'Rolling Back',
  };
  return labels[state] || state;
}

function getStateColor(state: string): { bg: string; text: string; border: string } {
  switch (state) {
    case 'Idle':
      return { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' };
    case 'WaitingForQuorum':
      return { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' };
    case 'Available':
      return { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/30' };
    case 'Downloading':
    case 'Verifying':
    case 'PreflightCheck':
      return { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' };
    case 'ReadyToRestart':
    case 'RestartScheduled':
      return { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' };
    case 'Error':
      return { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' };
    case 'RollingBack':
      return { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30' };
    default:
      return { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500/30' };
  }
}

function getStateIcon(state: string) {
  switch (state) {
    case 'Idle':
    case 'Disabled':
      return <CheckCircle className="w-4 h-4" />;
    case 'WaitingForQuorum':
      return <Clock className="w-4 h-4" />;
    case 'Available':
      return <Download className="w-4 h-4" />;
    case 'Downloading':
    case 'Verifying':
    case 'PreflightCheck':
      return <Loader2 className="w-4 h-4 animate-spin" />;
    case 'ReadyToRestart':
    case 'RestartScheduled':
      return <ArrowUpCircle className="w-4 h-4" />;
    case 'Error':
      return <AlertTriangle className="w-4 h-4" />;
    case 'RollingBack':
      return <RefreshCw className="w-4 h-4 animate-spin" />;
    default:
      return <Settings className="w-4 h-4" />;
  }
}

function UpdatesTab({ autoUpdateStatus, updateInfo, onToggle, onSetEmail }: {
  autoUpdateStatus: AutoUpdateStatus | null;
  updateInfo: NodeUpdateInfo | null;
  onToggle: () => Promise<void>;
  onSetEmail: (email: string | null) => Promise<void>;
}) {
  const [toggling, setToggling] = useState(false);
  const [emailInput, setEmailInput] = useState('');
  const [emailSaving, setEmailSaving] = useState(false);
  const [emailDirty, setEmailDirty] = useState(false);

  const handleToggle = async () => {
    setToggling(true);
    await onToggle();
    setToggling(false);
  };

  const handleSaveEmail = async () => {
    setEmailSaving(true);
    await onSetEmail(emailInput.trim() || null);
    setEmailSaving(false);
    setEmailDirty(false);
  };

  const handleClearEmail = async () => {
    setEmailSaving(true);
    await onSetEmail(null);
    setEmailInput('');
    setEmailSaving(false);
    setEmailDirty(false);
  };

  if (!autoUpdateStatus) {
    return (
      <div className="text-center py-8">
        <ArrowUpCircle className="w-8 h-8 text-slate-600 mx-auto mb-3" />
        <p className="text-slate-400">Update status not available</p>
        <p className="text-xs text-slate-500 mt-1">The node may not support the auto-update API yet</p>
      </div>
    );
  }

  const stateData = autoUpdateStatus.state;
  const stateName = stateData.state;
  const color = getStateColor(stateName);

  return (
    <div className="space-y-4">
      {/* Version + auto-update status with toggle */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Server className="w-4 h-4 text-blue-400" />}
          label="Current Version"
          value={`v${autoUpdateStatus.current_version}`}
        />
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {autoUpdateStatus.auto_update_enabled
                ? <CheckCircle className="w-4 h-4 text-green-400" />
                : <AlertTriangle className="w-4 h-4 text-amber-400" />}
              <span className="text-xs text-slate-400 uppercase tracking-wide">Auto-Update</span>
            </div>
            <button
              onClick={handleToggle}
              disabled={toggling}
              className={`relative w-11 h-6 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${
                autoUpdateStatus.auto_update_enabled
                  ? 'bg-green-500/80'
                  : 'bg-slate-600'
              } ${toggling ? 'opacity-50' : ''}`}
              title={autoUpdateStatus.auto_update_enabled ? 'Disable auto-update' : 'Enable auto-update'}
            >
              <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                autoUpdateStatus.auto_update_enabled ? 'translate-x-5' : 'translate-x-0'
              }`} />
            </button>
          </div>
          <div className="text-xl font-bold text-white">
            {autoUpdateStatus.auto_update_enabled ? 'Enabled' : 'Disabled'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {autoUpdateStatus.auto_update_enabled ? 'Will apply updates automatically' : 'Notification only'}
          </div>
        </div>
      </div>

      {/* State badge */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-slate-400">Update State</span>
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-full ${color.bg} ${color.text} border ${color.border}`}>
            {getStateIcon(stateName)}
            {getStateLabel(stateName)}
          </span>
        </div>

        {/* State-specific detail panels */}
        {stateName === 'WaitingForQuorum' && 'signers_so_far' in stateData && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-amber-400">Collecting signatures...</span>
              <span className="text-slate-400 font-mono">
                {(stateData as { signers_so_far: number; signers_needed: number }).signers_so_far} / {(stateData as { signers_so_far: number; signers_needed: number }).signers_needed} signers
              </span>
            </div>
            <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-yellow-400 rounded-full transition-all"
                style={{ width: `${Math.min(100, ((stateData as { signers_so_far: number; signers_needed: number }).signers_so_far / (stateData as { signers_so_far: number; signers_needed: number }).signers_needed) * 100)}%` }}
              />
            </div>
            {'version' in stateData && (
              <div className="text-xs text-slate-500">Target version: v{(stateData as { version: string }).version}</div>
            )}
          </div>
        )}

        {stateName === 'Downloading' && 'progress_percent' in stateData && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-blue-400">Downloading binary...</span>
              <span className="text-slate-400 font-mono">{(stateData as { progress_percent: number }).progress_percent}%</span>
            </div>
            <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full transition-all"
                style={{ width: `${(stateData as { progress_percent: number }).progress_percent}%` }}
              />
            </div>
            {'version' in stateData && (
              <div className="text-xs text-slate-500">Downloading v{(stateData as { version: string }).version}</div>
            )}
          </div>
        )}

        {stateName === 'Available' && 'download_url' in stateData && (
          <div className="mt-3">
            <a
              href={(stateData as { download_url: string }).download_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-cyan-600 to-blue-600 rounded-lg hover:from-cyan-500 hover:to-blue-500 transition-all"
            >
              <Download className="w-4 h-4" />
              Download v{(stateData as { version: string }).version}
            </a>
            <div className="text-xs text-slate-500 mt-2 text-center">
              Auto-update is disabled. Download and apply manually.
            </div>
          </div>
        )}

        {stateName === 'RestartScheduled' && 'restart_in_secs' in stateData && (
          <div className="mt-3 bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
            <div className="flex items-center gap-2 text-purple-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">Restart in {(stateData as { restart_in_secs: number }).restart_in_secs}s</span>
            </div>
            {'version' in stateData && (
              <div className="text-xs text-purple-300/60 mt-1">Upgrading to v{(stateData as { version: string }).version}</div>
            )}
          </div>
        )}

        {stateName === 'Error' && 'message' in stateData && (
          <div className="mt-3 bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
              <div>
                <div className="text-sm text-red-400 break-all">{(stateData as { message: string }).message}</div>
                {'retry_count' in stateData && (stateData as { retry_count: number }).retry_count > 0 && (
                  <div className="text-xs text-red-300/60 mt-1">Retries: {(stateData as { retry_count: number }).retry_count}</div>
                )}
              </div>
            </div>
          </div>
        )}

        {stateName === 'RollingBack' && 'reason' in stateData && (
          <div className="mt-3 bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <RefreshCw className="w-4 h-4 text-orange-400 mt-0.5 shrink-0 animate-spin" />
              <div>
                <div className="text-sm text-orange-400">Rolling back...</div>
                <div className="text-xs text-orange-300/60 mt-1">{(stateData as { reason: string }).reason}</div>
              </div>
            </div>
          </div>
        )}

        {(stateName === 'Verifying' || stateName === 'PreflightCheck') && (
          <div className="mt-3 flex items-center gap-2 text-blue-400 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            {stateName === 'Verifying' ? 'Verifying checksums (SHA-256 + BLAKE3)...' : 'Running preflight check on new binary...'}
          </div>
        )}

        {stateName === 'ReadyToRestart' && (
          <div className="mt-3 flex items-center gap-2 text-purple-400 text-sm">
            <ArrowUpCircle className="w-4 h-4" />
            Binary verified and ready. Restart pending.
          </div>
        )}
      </div>

      {/* Version comparison from updateInfo */}
      {updateInfo && updateInfo.latest_version && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
            <Globe className="w-4 h-4 text-cyan-400" /> Version Comparison
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Running</span>
              <span className="text-xs font-mono text-white">v{updateInfo.current_version}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Latest</span>
              <span className={`text-xs font-mono ${updateInfo.update_available ? 'text-amber-400' : 'text-green-400'}`}>
                v{updateInfo.latest_version}
              </span>
            </div>
            {!updateInfo.update_available && (
              <div className="flex items-center gap-2 text-xs text-green-400 mt-1">
                <CheckCircle className="w-3.5 h-3.5" />
                Node is up to date
              </div>
            )}
          </div>
        </div>
      )}

      {/* Email notifications */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Globe className="w-4 h-4 text-blue-400" /> Email Notifications
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Receive an email from system@quillon.xyz when a new node version is available.
        </p>
        <div className="flex items-center gap-2">
          <input
            type="email"
            placeholder={autoUpdateStatus.notification_email || 'admin@example.com'}
            value={emailDirty ? emailInput : (autoUpdateStatus.notification_email || '')}
            onChange={e => { setEmailInput(e.target.value); setEmailDirty(true); }}
            className="flex-1 px-3 py-2 bg-slate-700/50 border border-slate-600/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          {(emailDirty || (!emailDirty && !autoUpdateStatus.notification_email)) && (
            <button
              onClick={handleSaveEmail}
              disabled={emailSaving}
              className="px-3 py-2 text-xs font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {emailSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Save'}
            </button>
          )}
          {autoUpdateStatus.notification_email && !emailDirty && (
            <button
              onClick={handleClearEmail}
              disabled={emailSaving}
              className="px-3 py-2 text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg hover:bg-red-500/20 disabled:opacity-50 transition-colors"
            >
              {emailSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Remove'}
            </button>
          )}
        </div>
        {autoUpdateStatus.notification_email && !emailDirty && (
          <div className="flex items-center gap-2 mt-2 text-xs text-green-400">
            <CheckCircle className="w-3.5 h-3.5" />
            Notifications will be sent to {autoUpdateStatus.notification_email}
          </div>
        )}
      </div>

      {/* Info footer */}
      <div className="bg-slate-800/30 border border-slate-700/30 rounded-lg p-3">
        <div className="text-xs text-slate-500 space-y-1">
          <div className="flex justify-between">
            <span>Quorum requirement:</span>
            <span className="text-slate-400">2-of-3 trusted bootstrap signers</span>
          </div>
          <div className="flex justify-between">
            <span>Verification:</span>
            <span className="text-slate-400">Ed25519 + SHA-256 + BLAKE3</span>
          </div>
          <div className="flex justify-between">
            <span>Safety gates:</span>
            <span className="text-slate-400">Min peers, sync check, preflight, rollback watchdog</span>
          </div>
        </div>
      </div>
    </div>
  );
}
