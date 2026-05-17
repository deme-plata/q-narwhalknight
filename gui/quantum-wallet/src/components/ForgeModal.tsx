import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Cpu, Package, Truck, Check, AlertTriangle, Loader2, RefreshCw, Flame, Server, Zap, Thermometer } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface ForgeRedemption {
  redemption_id: string;
  buyer_wallet: string;
  shipping_name: string;
  shipping_address: string;
  city: string;
  state_province: string;
  zip: string;
  country: string;
  phone: string;
  email: string;
  cpu_config: string;
  gpu_config: string;
  cooling_type: string;
  ram_gb: number;
  storage_config: string;
  nic_config: string;
  chassis_color: string;
  quantity: number;
  status: string;
  tracking_number: string | null;
  serial_number: string | null;
  machine_id: string | null;
  attestation_pubkey: string | null;
  created_at: number;
  fulfilled_at: number | null;
}

interface ForgeStats {
  total_supply: number;
  circulating: number;
  burned: number;
  remaining: number;
  redemptions: {
    total_orders: number;
    total_machines_redeemed: number;
    pending: number;
    configured: number;
    assembling: number;
    testing: number;
    shipped: number;
    delivered: number;
  };
  fleet_stats: {
    total_cores_ordered: number;
    epyc_configurations: number;
    xeon_configurations: number;
    gpu_equipped: number;
  };
}

interface ForgeModalProps {
  isOpen: boolean;
  onClose: () => void;
  isAdmin: boolean;
  forgeBalance: number;
}

const CPU_OPTIONS = [
  { value: 'epyc-9755-dual', label: 'AMD EPYC 9755 Dual (256 cores)', cores: 256, color: '#E84D1A' },
  { value: 'epyc-9654-dual', label: 'AMD EPYC 9654 Dual (192 cores)', cores: 192, color: '#E84D1A' },
  { value: 'xeon-w9-3595x-dual', label: 'Intel Xeon w9-3595X Dual (120 cores)', cores: 120, color: '#0068B5' },
];

const GPU_OPTIONS = [
  { value: 'none', label: 'No GPU (CPU Mining Only)', vram: '—', price: '$0' },
  { value: 'rtx-5090-dual', label: '2× NVIDIA RTX 5090', vram: '64 GB', price: '+$6K' },
  { value: 'rtx-5090-quad', label: '4× NVIDIA RTX 5090', vram: '128 GB', price: '+$10K' },
  { value: 'a100-dual', label: '2× NVIDIA A100 80GB', vram: '160 GB', price: '+$13.5K' },
  { value: 'l40-quad', label: '4× NVIDIA L40 48GB', vram: '192 GB', price: '+$23.5K' },
];

const COOLING_OPTIONS = [
  { value: 'liquid-copper', label: 'Copper Liquid Cooling', desc: 'Signature copper hardline (default)' },
  { value: 'liquid-black', label: 'Black Nickel Cooling', desc: 'Stealth black nickel tubing (+$100)' },
];

const CHASSIS_OPTIONS = [
  { value: 'titanium-copper', label: 'Titanium + Copper', color: '#B87333', accent: '#71797E' },
  { value: 'titanium-black', label: 'Titanium + DLC Black', color: '#1a1a1a', accent: '#333333' },
  { value: 'titanium-natural', label: 'Raw Titanium', color: '#71797E', accent: '#A9A9A9' },
];

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-yellow-400',
  configured: 'text-blue-400',
  assembling: 'text-cyan-400',
  testing: 'text-purple-400',
  shipped: 'text-orange-400',
  delivered: 'text-green-400',
};

const STATUS_ICONS: Record<string, typeof Package> = {
  pending: AlertTriangle,
  configured: Cpu,
  assembling: Server,
  testing: Zap,
  shipped: Truck,
  delivered: Check,
};

export default function ForgeModal({ isOpen, onClose, isAdmin, forgeBalance }: ForgeModalProps) {
  const [activeTab, setActiveTab] = useState<'info' | 'redeem' | 'orders' | 'admin'>(isAdmin ? 'admin' : 'info');
  const [stats, setStats] = useState<ForgeStats | null>(null);
  const [redemptions, setRedemptions] = useState<ForgeRedemption[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Redeem form state
  const [shippingName, setShippingName] = useState('');
  const [shippingAddress, setShippingAddress] = useState('');
  const [city, setCity] = useState('');
  const [stateProvince, setStateProvince] = useState('');
  const [zip, setZip] = useState('');
  const [country, setCountry] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [cpuConfig, setCpuConfig] = useState('epyc-9755-dual');
  const [gpuConfig, setGpuConfig] = useState('none');
  const [coolingType, setCoolingType] = useState('liquid-copper');
  const [ramGb, setRamGb] = useState(512);
  const [chassisColor, setChassisColor] = useState('titanium-copper');
  const [confirmBurn, setConfirmBurn] = useState(false);

  // Fulfill form state (admin)
  const [fulfillId, setFulfillId] = useState('');
  const [fulfillTracking, setFulfillTracking] = useState('');
  const [fulfillSerial, setFulfillSerial] = useState('');
  const [fulfillMachineId, setFulfillMachineId] = useState('');
  const [fulfillStatus, setFulfillStatus] = useState('configured');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, redemptionsRes] = await Promise.all([
        qnkAPI.getForgeStats(),
        qnkAPI.getForgeRedemptions(),
      ]);
      if (statsRes.success && statsRes.data) setStats(statsRes.data);
      if (redemptionsRes.success && redemptionsRes.data) {
        setRedemptions(redemptionsRes.data.redemptions || []);
      }
    } catch (err) {
      console.error('[ForgeModal] Failed to fetch data:', err);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (isOpen) fetchData();
  }, [isOpen, fetchData]);

  const handleRedeem = async () => {
    if (!confirmBurn) {
      setMessage({ type: 'error', text: 'Please confirm that you understand 1 FORGE token will be burned.' });
      return;
    }
    if (!shippingName || !shippingAddress || !city || !country || !email) {
      setMessage({ type: 'error', text: 'Please fill in all required shipping fields.' });
      return;
    }

    setSubmitting(true);
    setMessage(null);
    try {
      const res = await qnkAPI.redeemForge({
        shipping_name: shippingName,
        shipping_address: shippingAddress,
        city,
        state_province: stateProvince,
        zip,
        country,
        phone,
        email,
        cpu_config: cpuConfig,
        gpu_config: gpuConfig,
        cooling_type: coolingType,
        ram_gb: ramGb,
        chassis_color: chassisColor,
        quantity: 1,
      });
      if (res.success) {
        setMessage({ type: 'success', text: res.data?.message || 'Your Quillon Forge order has been placed!' });
        setShippingName(''); setShippingAddress(''); setCity(''); setStateProvince('');
        setZip(''); setCountry(''); setPhone(''); setEmail(''); setConfirmBurn(false);
        fetchData();
      } else {
        setMessage({ type: 'error', text: res.error || 'Redemption failed.' });
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Network error.' });
    }
    setSubmitting(false);
  };

  const handleFulfill = async () => {
    if (!fulfillId) return;
    setSubmitting(true);
    setMessage(null);
    try {
      const res = await qnkAPI.fulfillForgeRedemption({
        redemption_id: fulfillId,
        tracking_number: fulfillTracking || undefined,
        serial_number: fulfillSerial || undefined,
        machine_id: fulfillMachineId || undefined,
        status: fulfillStatus,
      });
      if (res.success) {
        setMessage({ type: 'success', text: 'Forge redemption updated successfully.' });
        setFulfillId(''); setFulfillTracking(''); setFulfillSerial(''); setFulfillMachineId('');
        fetchData();
      } else {
        setMessage({ type: 'error', text: res.error || 'Update failed.' });
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Network error.' });
    }
    setSubmitting(false);
  };

  if (!isOpen) return null;

  const copperGrad = 'linear-gradient(135deg, #B87333 0%, #D4944A 50%, #8B5E34 100%)';
  const copperBorder = 'rgba(184, 115, 51, 0.3)';

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 overflow-y-auto"
        style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}
        onClick={onClose}
      >
        <div className="flex min-h-full items-center justify-center p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border"
          style={{
            background: 'linear-gradient(135deg, #1a1008 0%, #2a1a0e 50%, #0f0a05 100%)',
            borderColor: copperBorder,
            boxShadow: '0 0 60px rgba(184, 115, 51, 0.15)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b" style={{ borderColor: 'rgba(184, 115, 51, 0.15)' }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: copperGrad }}>
                  <Cpu className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">QUILLON FORGE</h2>
                  <p className="text-sm text-gray-400">
                    {isAdmin ? 'Admin Dashboard' : '256-Core Post-Quantum Mining Machine'}
                  </p>
                </div>
              </div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mt-4">
              {isAdmin && (
                <button
                  onClick={() => setActiveTab('admin')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === 'admin'
                      ? 'bg-orange-700/30 text-orange-300 border border-orange-500/40'
                      : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                  }`}
                >
                  Admin
                </button>
              )}
              <button
                onClick={() => setActiveTab('info')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'info'
                    ? 'bg-orange-700/30 text-orange-300 border border-orange-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                Specs
              </button>
              <button
                onClick={() => setActiveTab('redeem')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'redeem'
                    ? 'bg-orange-700/30 text-orange-300 border border-orange-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                Redeem Machine
              </button>
              <button
                onClick={() => setActiveTab('orders')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'orders'
                    ? 'bg-orange-700/30 text-orange-300 border border-orange-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                My Orders
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {message && (
              <div className={`mb-4 p-3 rounded-lg text-sm ${
                message.type === 'success' ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-red-500/20 text-red-300 border border-red-500/30'
              }`}>
                {message.text}
              </div>
            )}

            {/* Admin Tab */}
            {activeTab === 'admin' && isAdmin && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Supply Overview</h3>
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'Total Supply', value: stats?.total_supply ?? 500, color: 'text-white' },
                      { label: 'Circulating', value: stats?.circulating ?? '—', color: 'text-blue-400' },
                      { label: 'Burned', value: stats?.burned ?? 0, color: 'text-orange-400' },
                      { label: 'Remaining', value: stats?.remaining ?? '—', color: 'text-green-400' },
                    ].map((stat) => (
                      <div key={stat.label} className="p-3 rounded-xl" style={{ background: 'rgba(184, 115, 51, 0.08)', border: '1px solid rgba(184, 115, 51, 0.15)' }}>
                        <p className="text-xs text-gray-400">{stat.label}</p>
                        <p className={`text-xl font-bold ${stat.color}`}>{stat.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Fleet Stats */}
                {stats?.fleet_stats && (
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'Total Cores', value: stats.fleet_stats.total_cores_ordered, color: 'text-cyan-400' },
                      { label: 'EPYC Configs', value: stats.fleet_stats.epyc_configurations, color: 'text-red-400' },
                      { label: 'Xeon Configs', value: stats.fleet_stats.xeon_configurations, color: 'text-blue-400' },
                      { label: 'GPU Equipped', value: stats.fleet_stats.gpu_equipped, color: 'text-green-400' },
                    ].map((stat) => (
                      <div key={stat.label} className="p-3 rounded-xl" style={{ background: 'rgba(184, 115, 51, 0.05)', border: '1px solid rgba(184, 115, 51, 0.1)' }}>
                        <p className="text-xs text-gray-500">{stat.label}</p>
                        <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Order Queue */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-white">Order Queue</h3>
                    <button onClick={fetchData} className="p-1.5 rounded-lg hover:bg-white/10">
                      <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                  </div>
                  {redemptions.length === 0 ? (
                    <p className="text-gray-500 text-sm">No orders yet.</p>
                  ) : (
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {redemptions.map((r) => {
                        const StatusIcon = STATUS_ICONS[r.status] || Package;
                        return (
                          <div key={r.redemption_id} className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-mono text-orange-300">{r.redemption_id}</span>
                              <span className={`text-xs font-medium flex items-center gap-1 ${STATUS_COLORS[r.status] || 'text-gray-400'}`}>
                                <StatusIcon className="w-3 h-3" />
                                {r.status.toUpperCase()}
                              </span>
                            </div>
                            <p className="text-sm text-white">{r.shipping_name} — {r.cpu_config} / {r.gpu_config} x{r.quantity}</p>
                            <p className="text-xs text-gray-500">{r.city}, {r.country}</p>
                            {r.machine_id && <p className="text-xs text-cyan-400 mt-1">Machine: {r.machine_id}</p>}
                            {r.tracking_number && <p className="text-xs text-blue-400">Tracking: {r.tracking_number}</p>}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Fulfill Form */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Fulfill Order</h3>
                  <div className="space-y-3">
                    <select value={fulfillId} onChange={(e) => setFulfillId(e.target.value)}
                      className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none">
                      <option value="">Select order...</option>
                      {redemptions.filter(r => r.status !== 'delivered').map(r => (
                        <option key={r.redemption_id} value={r.redemption_id}>
                          {r.redemption_id} — {r.shipping_name} ({r.status})
                        </option>
                      ))}
                    </select>
                    <div className="grid grid-cols-2 gap-3">
                      <input value={fulfillTracking} onChange={(e) => setFulfillTracking(e.target.value)}
                        placeholder="Tracking number" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      <input value={fulfillSerial} onChange={(e) => setFulfillSerial(e.target.value)}
                        placeholder="Serial number" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                    </div>
                    <input value={fulfillMachineId} onChange={(e) => setFulfillMachineId(e.target.value)}
                      placeholder="Machine ID (e.g. QF-2026-00001)" className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                    <select value={fulfillStatus} onChange={(e) => setFulfillStatus(e.target.value)}
                      className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none">
                      <option value="configured">Configured</option>
                      <option value="assembling">Assembling</option>
                      <option value="testing">Testing</option>
                      <option value="shipped">Shipped</option>
                      <option value="delivered">Delivered</option>
                    </select>
                    <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                      onClick={handleFulfill} disabled={!fulfillId || submitting}
                      className="w-full py-3 rounded-xl font-medium text-white disabled:opacity-50"
                      style={{ background: copperGrad }}>
                      {submitting ? <Loader2 className="w-5 h-5 mx-auto animate-spin" /> : 'Update Order'}
                    </motion.button>
                  </div>
                </div>
              </div>
            )}

            {/* Specs Tab */}
            {activeTab === 'info' && (
              <div className="space-y-5">
                <div className="text-center">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-2xl flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, #2a1a0e, #3d2818)', border: '2px solid rgba(184, 115, 51, 0.3)' }}>
                    <Cpu className="w-12 h-12 text-orange-400" />
                  </div>
                  <h3 className="text-2xl font-bold text-white">Quillon Forge</h3>
                  <p className="text-gray-400 mt-1">256-Core Post-Quantum Mining Machine</p>
                  <p className="text-xs text-orange-400/80 mt-1 italic">Plug and Play — Power on, connect Ethernet, mine in 60 seconds</p>
                </div>

                <div className="p-4 rounded-xl" style={{ background: 'rgba(184, 115, 51, 0.08)', border: '1px solid rgba(184, 115, 51, 0.15)' }}>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    The Quillon Forge is a purpose-built 4U rackmount mining appliance for Q-NarwhalKnight. Dual AMD EPYC 9755 CPUs with
                    AVX-512 for post-quantum cryptography, copper liquid cooling with sapphire viewport, and optional GPU for AI Proof of Inference.
                    Each FORGE token represents one physical machine — burn to redeem.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'CPU', value: '2× EPYC 9755 (256 cores)' },
                    { label: 'RAM', value: '512 GB DDR5 ECC' },
                    { label: 'Storage', value: '2× 3.84TB NVMe RAID-1' },
                    { label: 'Network', value: '100 GbE (Mellanox CX-7)' },
                    { label: 'Cooling', value: 'Copper liquid cooling' },
                    { label: 'PSU', value: '2× 1600W 80+ Titanium' },
                    { label: 'Chassis', value: '4U titanium-anodized' },
                    { label: 'Boot Time', value: '60 seconds to mining' },
                  ].map((spec) => (
                    <div key={spec.label} className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <p className="text-xs text-gray-500">{spec.label}</p>
                      <p className="text-sm text-white font-medium">{spec.value}</p>
                    </div>
                  ))}
                </div>

                <div className="text-center">
                  <p className="text-lg font-bold text-orange-300">Your FORGE Balance: {forgeBalance}</p>
                  {forgeBalance > 0 && (
                    <button onClick={() => setActiveTab('redeem')}
                      className="mt-3 px-6 py-2.5 rounded-xl text-white font-medium" style={{ background: copperGrad }}>
                      Redeem Physical Machine
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Redeem Tab */}
            {activeTab === 'redeem' && (
              <div className="space-y-4">
                {forgeBalance < 1 ? (
                  <div className="text-center py-8">
                    <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-3" />
                    <p className="text-white font-medium">No FORGE Tokens</p>
                    <p className="text-gray-400 text-sm mt-1">You need at least 1 FORGE token to redeem a physical mining machine.</p>
                  </div>
                ) : (
                  <>
                    {/* Hardware Configuration */}
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Cpu className="w-5 h-5 text-orange-400" /> Hardware Configuration
                    </h3>

                    {/* CPU Selection */}
                    <div>
                      <p className="text-sm text-gray-400 mb-2">CPU Configuration</p>
                      <div className="space-y-2">
                        {CPU_OPTIONS.map((cpu) => (
                          <button key={cpu.value} onClick={() => setCpuConfig(cpu.value)}
                            className={`w-full p-3 rounded-xl text-left transition-all ${
                              cpuConfig === cpu.value ? 'ring-2 ring-orange-400' : 'hover:bg-white/5'
                            }`}
                            style={{ background: cpuConfig === cpu.value ? 'rgba(184, 115, 51, 0.15)' : 'rgba(255,255,255,0.03)',
                              border: cpuConfig === cpu.value ? '1px solid rgba(184, 115, 51, 0.4)' : '1px solid rgba(255,255,255,0.06)' }}>
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-white font-medium">{cpu.label}</span>
                              <span className="text-xs text-orange-300">{cpu.cores} cores</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* GPU Selection */}
                    <div>
                      <p className="text-sm text-gray-400 mb-2">GPU Configuration (Optional — for AI Inference)</p>
                      <div className="space-y-2">
                        {GPU_OPTIONS.map((gpu) => (
                          <button key={gpu.value} onClick={() => setGpuConfig(gpu.value)}
                            className={`w-full p-3 rounded-xl text-left transition-all ${
                              gpuConfig === gpu.value ? 'ring-2 ring-orange-400' : 'hover:bg-white/5'
                            }`}
                            style={{ background: gpuConfig === gpu.value ? 'rgba(184, 115, 51, 0.15)' : 'rgba(255,255,255,0.03)',
                              border: gpuConfig === gpu.value ? '1px solid rgba(184, 115, 51, 0.4)' : '1px solid rgba(255,255,255,0.06)' }}>
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-white">{gpu.label}</span>
                              <div className="text-right">
                                <span className="text-xs text-gray-400 block">{gpu.vram} VRAM</span>
                                <span className="text-xs text-orange-300">{gpu.price}</span>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* RAM */}
                    <div>
                      <p className="text-sm text-gray-400 mb-2">RAM</p>
                      <div className="grid grid-cols-3 gap-2">
                        {[512, 1024, 2048].map((gb) => (
                          <button key={gb} onClick={() => setRamGb(gb)}
                            className={`p-3 rounded-xl text-center transition-all ${ramGb === gb ? 'ring-2 ring-orange-400' : 'hover:bg-white/5'}`}
                            style={{ background: ramGb === gb ? 'rgba(184, 115, 51, 0.15)' : 'rgba(255,255,255,0.03)',
                              border: ramGb === gb ? '1px solid rgba(184, 115, 51, 0.4)' : '1px solid rgba(255,255,255,0.06)' }}>
                            <p className="text-white font-bold">{gb} GB</p>
                            <p className="text-xs text-gray-500">DDR5 ECC</p>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Chassis Color */}
                    <div>
                      <p className="text-sm text-gray-400 mb-2">Chassis Color</p>
                      <div className="grid grid-cols-3 gap-2">
                        {CHASSIS_OPTIONS.map((v) => (
                          <button key={v.value} onClick={() => setChassisColor(v.value)}
                            className={`p-3 rounded-xl text-center transition-all ${
                              chassisColor === v.value ? 'ring-2 ring-orange-400 ring-offset-2 ring-offset-transparent' : 'hover:bg-white/5'
                            }`}
                            style={{ background: `linear-gradient(135deg, ${v.color}, ${v.accent})`,
                              border: chassisColor === v.value ? '2px solid rgba(184, 115, 51, 0.6)' : '1px solid rgba(255,255,255,0.1)' }}>
                            <p className="text-xs font-medium text-white drop-shadow-lg">{v.label}</p>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Shipping Info */}
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2 pt-2">
                      <Truck className="w-5 h-5 text-orange-400" /> Shipping Information
                    </h3>
                    <div className="space-y-3">
                      <input value={shippingName} onChange={(e) => setShippingName(e.target.value)}
                        placeholder="Full Name *" className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      <input value={shippingAddress} onChange={(e) => setShippingAddress(e.target.value)}
                        placeholder="Street Address *" className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      <div className="grid grid-cols-2 gap-3">
                        <input value={city} onChange={(e) => setCity(e.target.value)}
                          placeholder="City *" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                        <input value={stateProvince} onChange={(e) => setStateProvince(e.target.value)}
                          placeholder="State/Province" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <input value={zip} onChange={(e) => setZip(e.target.value)}
                          placeholder="ZIP / Postal Code" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                        <input value={country} onChange={(e) => setCountry(e.target.value)}
                          placeholder="Country *" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <input value={phone} onChange={(e) => setPhone(e.target.value)}
                          placeholder="Phone" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                        <input value={email} onChange={(e) => setEmail(e.target.value)}
                          placeholder="Email *" type="email" className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-orange-500/50 outline-none" />
                      </div>
                    </div>

                    {/* Burn Confirmation */}
                    <label className="flex items-start gap-3 p-3 rounded-xl cursor-pointer" style={{ background: 'rgba(234, 88, 12, 0.1)', border: '1px solid rgba(234, 88, 12, 0.2)' }}>
                      <input type="checkbox" checked={confirmBurn} onChange={(e) => setConfirmBurn(e.target.checked)} className="mt-0.5 accent-orange-500" />
                      <span className="text-sm text-orange-300">
                        I understand that 1 FORGE token will be permanently burned to order a physical Quillon Forge mining machine.
                        Hardware configuration cannot be changed after submission. This action cannot be undone.
                      </span>
                    </label>

                    <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                      onClick={handleRedeem} disabled={submitting || !confirmBurn}
                      className="w-full py-3.5 rounded-xl font-bold text-white disabled:opacity-50 flex items-center justify-center gap-2"
                      style={{ background: confirmBurn ? copperGrad : 'rgba(100,100,100,0.3)' }}>
                      {submitting ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <>
                          <Flame className="w-5 h-5" />
                          Burn 1 FORGE & Order Machine
                        </>
                      )}
                    </motion.button>
                  </>
                )}
              </div>
            )}

            {/* Orders Tab */}
            {activeTab === 'orders' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">My Orders</h3>
                  <button onClick={fetchData} className="p-1.5 rounded-lg hover:bg-white/10">
                    <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
                {loading ? (
                  <div className="text-center py-8">
                    <Loader2 className="w-8 h-8 text-orange-400 animate-spin mx-auto" />
                  </div>
                ) : redemptions.length === 0 ? (
                  <div className="text-center py-8">
                    <Server className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">No orders yet.</p>
                    {forgeBalance > 0 && (
                      <button onClick={() => setActiveTab('redeem')} className="mt-3 text-orange-400 text-sm hover:text-orange-300">
                        Configure & order a machine →
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {redemptions.map((r) => {
                      const StatusIcon = STATUS_ICONS[r.status] || Package;
                      return (
                        <div key={r.redemption_id} className="p-4 rounded-xl" style={{ background: 'rgba(184, 115, 51, 0.05)', border: '1px solid rgba(184, 115, 51, 0.12)' }}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-mono text-orange-300">{r.redemption_id}</span>
                            <span className={`text-sm font-medium flex items-center gap-1.5 ${STATUS_COLORS[r.status] || 'text-gray-400'}`}>
                              <StatusIcon className="w-4 h-4" />
                              {r.status.charAt(0).toUpperCase() + r.status.slice(1)}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <p className="text-gray-400">CPU: <span className="text-white">{r.cpu_config}</span></p>
                            <p className="text-gray-400">GPU: <span className="text-white">{r.gpu_config}</span></p>
                            <p className="text-gray-400">RAM: <span className="text-white">{r.ram_gb} GB</span></p>
                            <p className="text-gray-400">Chassis: <span className="text-white">{r.chassis_color}</span></p>
                            <p className="text-gray-400">Ship to: <span className="text-white">{r.shipping_name}</span></p>
                            <p className="text-gray-400">Date: <span className="text-white">{new Date(r.created_at * 1000).toLocaleDateString()}</span></p>
                          </div>
                          {r.machine_id && (
                            <p className="text-sm text-cyan-400 mt-2">
                              <Cpu className="w-3.5 h-3.5 inline mr-1" />
                              Machine ID: {r.machine_id}
                            </p>
                          )}
                          {r.tracking_number && (
                            <p className="text-sm text-blue-400">
                              <Truck className="w-3.5 h-3.5 inline mr-1" />
                              Tracking: {r.tracking_number}
                            </p>
                          )}
                          {r.serial_number && (
                            <p className="text-sm text-green-400">
                              <Server className="w-3.5 h-3.5 inline mr-1" />
                              Serial: {r.serial_number}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </motion.div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
