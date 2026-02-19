// v7.3.1: Node Settings Modal — Admin-only modal for node configuration, OAuth2, fees, and updates
// Listens for 'open-node-settings' custom event from TopBar gear icon

import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Settings, Shield, Globe, Key, Trash2, RefreshCw, Clock, Server, Wifi, DollarSign, Download, ArrowUpCircle, CheckCircle } from 'lucide-react';

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

interface NodeUpdateInfo {
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  download_url: string | null;
}

type TabId = 'overview' | 'oauth2' | 'node' | 'fees';

function formatUptime(secs: number): string {
  const days = Math.floor(secs / 86400);
  const hours = Math.floor((secs % 86400) / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

function getAuthHeaders(): Record<string, string> {
  const wallet = localStorage.getItem('walletAddress') || '';
  return {
    'X-Wallet-Auth': wallet,
    'Authorization': `Bearer ${wallet}`,
    'Content-Type': 'application/json',
  };
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
  const [loading, setLoading] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
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
    if (!checkIsMasterWallet()) return;
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

  // Fetch data when modal opens
  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    Promise.all([fetchSettings(), fetchConsents(), fetchNodeInfo(), fetchOperatorFees(), fetchUpdateInfo()])
      .finally(() => setLoading(false));
  }, [isOpen, fetchSettings, fetchConsents, fetchNodeInfo, fetchOperatorFees, fetchUpdateInfo]);

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
    Promise.all([fetchSettings(), fetchConsents(), fetchNodeInfo(), fetchOperatorFees(), fetchUpdateInfo()])
      .finally(() => setLoading(false));
  };

  if (!isOpen) return null;

  const tabs: { id: TabId; label: string; icon: React.ReactNode; masterOnly?: boolean }[] = [
    { id: 'overview', label: 'Overview', icon: <Settings className="w-4 h-4" /> },
    { id: 'oauth2', label: 'OAuth2', icon: <Key className="w-4 h-4" /> },
    { id: 'node', label: 'Node', icon: <Server className="w-4 h-4" /> },
    { id: 'fees', label: 'Fees', icon: <DollarSign className="w-4 h-4" />, masterOnly: true },
  ];

  const visibleTabs = tabs.filter(t => !t.masterOnly || isMaster);

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
            className="relative w-full max-w-2xl mx-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 border border-blue-500/30 rounded-2xl shadow-2xl overflow-hidden"
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
                  <p className="text-xs text-slate-400">Admin configuration panel</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
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
            <div className="flex border-b border-slate-700/50">
              {visibleTabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-colors ${
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
            <div className="p-6 max-h-[60vh] overflow-y-auto">
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
                <OverviewTab settings={settings} syncPct={syncPct} />
              ) : activeTab === 'oauth2' ? (
                <OAuth2Tab
                  settings={settings}
                  consents={consents}
                  revoking={revoking}
                  onRevoke={handleRevoke}
                />
              ) : activeTab === 'fees' ? (
                <FeesTab
                  fees={operatorFees}
                  onUpdate={async (promille, bps) => {
                    try {
                      const body: Record<string, number> = {};
                      if (promille !== undefined) body.node_operator_fee_promille = promille;
                      if (bps !== undefined) body.dex_protocol_fee_bps = bps;
                      const res = await fetch('/api/v1/admin/operator-fees', {
                        method: 'POST',
                        headers: getAuthHeaders(),
                        body: JSON.stringify(body),
                      });
                      if (res.ok) setOperatorFees(await res.json());
                    } catch { /* ignore */ }
                  }}
                />
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

// -- Overview Tab --

function StatCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub?: string }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );
}

function OverviewTab({ settings, syncPct }: { settings: AdminSettings | null; syncPct: number }) {
  if (!settings) return <p className="text-slate-400">No data available</p>;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Shield className="w-4 h-4 text-emerald-400" />}
          label="Admin Wallet"
          value={settings.admin_wallet}
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

      {/* Sync progress */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Sync Progress</span>
          <span className="text-sm font-mono text-white">{syncPct.toFixed(1)}%</span>
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

// -- Fees Tab (Master Wallet Only) --

function FeesTab({ fees, onUpdate }: {
  fees: OperatorFees | null;
  onUpdate: (promille?: number, bps?: number) => Promise<void>;
}) {
  const [editPromille, setEditPromille] = useState('');
  const [editBps, setEditBps] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (fees) {
      setEditPromille(String(fees.node_operator_fee_promille));
      setEditBps(String(fees.dex_protocol_fee_bps));
    }
  }, [fees]);

  if (!fees) return <p className="text-slate-400">Fee data not available (master wallet only)</p>;

  const handleSave = async () => {
    setSaving(true);
    const promille = parseInt(editPromille);
    const bps = parseInt(editBps);
    const updates: { promille?: number; bps?: number } = {};
    if (!isNaN(promille) && promille !== fees.node_operator_fee_promille) updates.promille = promille;
    if (!isNaN(bps) && bps !== fees.dex_protocol_fee_bps) updates.bps = bps;
    if (updates.promille !== undefined || updates.bps !== undefined) {
      await onUpdate(updates.promille, updates.bps);
    }
    setSaving(false);
  };

  const promilleNum = parseInt(editPromille);
  const bpsNum = parseInt(editBps);
  const hasChanges = (
    (!isNaN(promilleNum) && promilleNum !== fees.node_operator_fee_promille) ||
    (!isNaN(bpsNum) && bpsNum !== fees.dex_protocol_fee_bps)
  );
  const promilleValid = !isNaN(promilleNum) && promilleNum >= 0 && promilleNum <= 500;
  const bpsValid = !isNaN(bpsNum) && bpsNum >= 0 && bpsNum <= 10;

  return (
    <div className="space-y-4">
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
          sub={fees.admin_wallet}
        />
      </div>

      {/* Node Operator Fee */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-amber-400" /> Node Operator Fee Share
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Percentage of collected transaction fees routed to the node operator wallet.
          The rest goes to the protocol treasury.
        </p>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <label className="text-xs text-slate-400 mb-1 block">Promille (0-500)</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={0}
                max={500}
                value={editPromille}
                onChange={e => setEditPromille(e.target.value)}
                className={`w-24 px-3 py-2 bg-slate-700/50 border rounded-lg text-sm text-white font-mono focus:outline-none focus:ring-1 ${
                  promilleValid ? 'border-slate-600/50 focus:ring-blue-500' : 'border-red-500/50 focus:ring-red-500'
                }`}
              />
              <span className="text-sm text-slate-400">
                = {promilleValid ? (promilleNum / 10).toFixed(1) : '?'}%
              </span>
            </div>
            {!promilleValid && <p className="text-xs text-red-400 mt-1">Must be 0-500 (0%-50%)</p>}
          </div>
        </div>
      </div>

      {/* DEX Protocol Fee */}
      <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-cyan-400" /> DEX Protocol Fee
        </h3>
        <p className="text-xs text-slate-500 mb-3">
          Protocol fee extracted from each DEX swap (in basis points).
          Split between treasury and operator based on the operator fee share above.
        </p>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <label className="text-xs text-slate-400 mb-1 block">Basis Points (0-10)</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={0}
                max={10}
                value={editBps}
                onChange={e => setEditBps(e.target.value)}
                className={`w-24 px-3 py-2 bg-slate-700/50 border rounded-lg text-sm text-white font-mono focus:outline-none focus:ring-1 ${
                  bpsValid ? 'border-slate-600/50 focus:ring-blue-500' : 'border-red-500/50 focus:ring-red-500'
                }`}
              />
              <span className="text-sm text-slate-400">
                = {bpsValid ? (bpsNum / 100).toFixed(2) : '?'}%
              </span>
            </div>
            {!bpsValid && <p className="text-xs text-red-400 mt-1">Must be 0-10 (0%-0.1%)</p>}
          </div>
        </div>
      </div>

      {/* Save button */}
      {hasChanges && promilleValid && bpsValid && (
        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 transition-all"
        >
          {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
          Save Fee Settings
        </button>
      )}

      {/* Current summary */}
      <div className="bg-slate-800/30 border border-slate-700/30 rounded-lg p-3">
        <div className="text-xs text-slate-500 space-y-1">
          <div className="flex justify-between">
            <span>Transaction Fee Split:</span>
            <span className="text-slate-400">
              {100 - (fees.node_operator_fee_promille / 10)}% treasury / {(fees.node_operator_fee_promille / 10)}% operator
            </span>
          </div>
          <div className="flex justify-between">
            <span>DEX Protocol Fee:</span>
            <span className="text-slate-400">{fees.dex_protocol_fee_percent} per swap</span>
          </div>
          <div className="flex justify-between">
            <span>LP Fee (unchanged):</span>
            <span className="text-slate-400">0.30% (stays in pool)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
