import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Shield, Package, Truck, Check, AlertTriangle, ChevronDown, Loader2, ExternalLink, RefreshCw } from 'lucide-react';
import { qnkAPI } from '../services/api';

interface VaultRedemption {
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
  color_variant: string;
  quantity: number;
  status: string;
  tracking_number: string | null;
  serial_number: string | null;
  created_at: number;
  fulfilled_at: number | null;
}

interface VaultStats {
  total_supply: number;
  circulating: number;
  burned: number;
  remaining: number;
  redemptions: {
    total_orders: number;
    total_devices_redeemed: number;
    pending: number;
    processing: number;
    shipped: number;
    delivered: number;
  };
}

interface VaultModalProps {
  isOpen: boolean;
  onClose: () => void;
  isAdmin: boolean;
  vaultBalance: number;
}

const COLOR_VARIANTS = [
  { name: 'Obsidian', color: '#1a1a2e', accent: '#16213e' },
  { name: 'Titanium', color: '#71797E', accent: '#A9A9A9' },
  { name: 'Rose Gold', color: '#B76E79', accent: '#E8C4C8' },
  { name: 'Stealth Black', color: '#0d0d0d', accent: '#333333' },
  { name: 'Arctic White', color: '#F0F0F0', accent: '#FFFFFF' },
  { name: 'Carbon Fiber', color: '#2C2C2C', accent: '#4A4A4A' },
];

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-yellow-400',
  processing: 'text-blue-400',
  shipped: 'text-purple-400',
  delivered: 'text-green-400',
};

const STATUS_ICONS: Record<string, typeof Package> = {
  pending: AlertTriangle,
  processing: Package,
  shipped: Truck,
  delivered: Check,
};

export default function VaultModal({ isOpen, onClose, isAdmin, vaultBalance }: VaultModalProps) {
  const [activeTab, setActiveTab] = useState<'info' | 'redeem' | 'orders' | 'admin'>(isAdmin ? 'admin' : 'info');
  const [stats, setStats] = useState<VaultStats | null>(null);
  const [redemptions, setRedemptions] = useState<VaultRedemption[]>([]);
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
  const [colorVariant, setColorVariant] = useState('Obsidian');
  const [confirmBurn, setConfirmBurn] = useState(false);

  // Fulfill form state (admin)
  const [fulfillId, setFulfillId] = useState('');
  const [fulfillTracking, setFulfillTracking] = useState('');
  const [fulfillSerial, setFulfillSerial] = useState('');
  const [fulfillStatus, setFulfillStatus] = useState('shipped');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, redemptionsRes] = await Promise.all([
        qnkAPI.getVaultTokenStats(),
        qnkAPI.getVaultRedemptions(),
      ]);
      if (statsRes.success && statsRes.data) setStats(statsRes.data);
      if (redemptionsRes.success && redemptionsRes.data) {
        setRedemptions(redemptionsRes.data.redemptions || []);
      }
    } catch (err) {
      console.error('[VaultModal] Failed to fetch data:', err);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (isOpen) fetchData();
  }, [isOpen, fetchData]);

  const handleRedeem = async () => {
    if (!confirmBurn) {
      setMessage({ type: 'error', text: 'Please confirm that you understand 1 VAULT token will be burned.' });
      return;
    }
    if (!shippingName || !shippingAddress || !city || !country || !email) {
      setMessage({ type: 'error', text: 'Please fill in all required shipping fields.' });
      return;
    }

    setSubmitting(true);
    setMessage(null);
    try {
      const res = await qnkAPI.redeemVault({
        shipping_name: shippingName,
        shipping_address: shippingAddress,
        city,
        state_province: stateProvince,
        zip,
        country,
        phone,
        email,
        color_variant: colorVariant,
        quantity: 1,
      });
      if (res.success) {
        setMessage({ type: 'success', text: res.data?.message || 'Redemption successful! Your order has been placed.' });
        // Reset form
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
      const res = await qnkAPI.fulfillVaultRedemption({
        redemption_id: fulfillId,
        tracking_number: fulfillTracking || undefined,
        serial_number: fulfillSerial || undefined,
        status: fulfillStatus,
      });
      if (res.success) {
        setMessage({ type: 'success', text: 'Redemption updated successfully.' });
        setFulfillId(''); setFulfillTracking(''); setFulfillSerial('');
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
            background: 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%)',
            borderColor: 'rgba(168, 130, 255, 0.3)',
            boxShadow: '0 0 60px rgba(168, 130, 255, 0.15)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b" style={{ borderColor: 'rgba(168, 130, 255, 0.15)' }}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, #a882ff 0%, #6c5ce7 100%)' }}>
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">QUILLON VAULT</h2>
                  <p className="text-sm text-gray-400">
                    {isAdmin ? 'Admin Dashboard' : 'Physical Hardware Wallet Token'}
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
                      ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
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
                    ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                Product Info
              </button>
              <button
                onClick={() => setActiveTab('redeem')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'redeem'
                    ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                Redeem Device
              </button>
              <button
                onClick={() => setActiveTab('orders')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'orders'
                    ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-white/5'
                }`}
              >
                My Orders
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {/* Message banner */}
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
                {/* Supply Stats */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Supply Overview</h3>
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'Total Supply', value: stats?.total_supply ?? 1000, color: 'text-white' },
                      { label: 'Circulating', value: stats?.circulating ?? '—', color: 'text-blue-400' },
                      { label: 'Burned', value: stats?.burned ?? 0, color: 'text-orange-400' },
                      { label: 'Remaining', value: stats?.remaining ?? '—', color: 'text-green-400' },
                    ].map((stat) => (
                      <div key={stat.label} className="p-3 rounded-xl" style={{ background: 'rgba(168, 130, 255, 0.08)', border: '1px solid rgba(168, 130, 255, 0.15)' }}>
                        <p className="text-xs text-gray-400">{stat.label}</p>
                        <p className={`text-xl font-bold ${stat.color}`}>{stat.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Redemption Stats */}
                {stats?.redemptions && (
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { label: 'Pending', value: stats.redemptions.pending, color: 'text-yellow-400' },
                      { label: 'Processing', value: stats.redemptions.processing, color: 'text-blue-400' },
                      { label: 'Shipped', value: stats.redemptions.shipped, color: 'text-purple-400' },
                      { label: 'Delivered', value: stats.redemptions.delivered, color: 'text-green-400' },
                    ].map((stat) => (
                      <div key={stat.label} className="p-3 rounded-xl" style={{ background: 'rgba(168, 130, 255, 0.05)', border: '1px solid rgba(168, 130, 255, 0.1)' }}>
                        <p className="text-xs text-gray-500">{stat.label}</p>
                        <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Redemption Queue */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-white">Redemption Queue</h3>
                    <button onClick={fetchData} className="p-1.5 rounded-lg hover:bg-white/10">
                      <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                  </div>
                  {redemptions.length === 0 ? (
                    <p className="text-gray-500 text-sm">No redemptions yet.</p>
                  ) : (
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                      {redemptions.map((r) => {
                        const StatusIcon = STATUS_ICONS[r.status] || Package;
                        return (
                          <div key={r.redemption_id} className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-mono text-purple-300">{r.redemption_id}</span>
                              <span className={`text-xs font-medium flex items-center gap-1 ${STATUS_COLORS[r.status] || 'text-gray-400'}`}>
                                <StatusIcon className="w-3 h-3" />
                                {r.status.toUpperCase()}
                              </span>
                            </div>
                            <p className="text-sm text-white">{r.shipping_name} — {r.color_variant} x{r.quantity}</p>
                            <p className="text-xs text-gray-500">{r.city}, {r.state_province} {r.zip}, {r.country}</p>
                            {r.tracking_number && <p className="text-xs text-blue-400 mt-1">Tracking: {r.tracking_number}</p>}
                            {r.serial_number && <p className="text-xs text-green-400">S/N: {r.serial_number}</p>}
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
                    <select
                      value={fulfillId}
                      onChange={(e) => setFulfillId(e.target.value)}
                      className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                    >
                      <option value="">Select redemption...</option>
                      {redemptions.filter(r => r.status !== 'delivered').map(r => (
                        <option key={r.redemption_id} value={r.redemption_id}>
                          {r.redemption_id} — {r.shipping_name} ({r.status})
                        </option>
                      ))}
                    </select>
                    <div className="grid grid-cols-2 gap-3">
                      <input
                        value={fulfillTracking}
                        onChange={(e) => setFulfillTracking(e.target.value)}
                        placeholder="Tracking number"
                        className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                      />
                      <input
                        value={fulfillSerial}
                        onChange={(e) => setFulfillSerial(e.target.value)}
                        placeholder="Serial number"
                        className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                      />
                    </div>
                    <select
                      value={fulfillStatus}
                      onChange={(e) => setFulfillStatus(e.target.value)}
                      className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                    >
                      <option value="processing">Processing</option>
                      <option value="shipped">Shipped</option>
                      <option value="delivered">Delivered</option>
                    </select>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleFulfill}
                      disabled={!fulfillId || submitting}
                      className="w-full py-3 rounded-xl font-medium text-white disabled:opacity-50"
                      style={{ background: 'linear-gradient(135deg, #6c5ce7, #a882ff)' }}
                    >
                      {submitting ? <Loader2 className="w-5 h-5 mx-auto animate-spin" /> : 'Update Redemption'}
                    </motion.button>
                  </div>
                </div>
              </div>
            )}

            {/* Product Info Tab */}
            {activeTab === 'info' && (
              <div className="space-y-5">
                <div className="text-center">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-2xl flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, #1a1a3e, #2d2d5e)', border: '2px solid rgba(168, 130, 255, 0.3)' }}>
                    <Shield className="w-12 h-12 text-purple-400" />
                  </div>
                  <h3 className="text-2xl font-bold text-white">Quillon Vault</h3>
                  <p className="text-gray-400 mt-1">Post-Quantum Hardware Wallet</p>
                </div>

                <div className="p-4 rounded-xl" style={{ background: 'rgba(168, 130, 255, 0.08)', border: '1px solid rgba(168, 130, 255, 0.15)' }}>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    The Quillon Vault is a 3.8mm ultra-thin titanium hardware wallet featuring a true Quantum Random Number Generator (QRNG),
                    Dilithium5 post-quantum signatures, and secure element isolation. Each VAULT token represents ownership of one physical device.
                    Burn your token to redeem the physical hardware wallet shipped worldwide.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Thickness', value: '3.8mm' },
                    { label: 'Material', value: 'Grade 5 Titanium' },
                    { label: 'QRNG', value: 'True Quantum RNG' },
                    { label: 'Signatures', value: 'Dilithium5 + Ed25519' },
                    { label: 'Secure Element', value: 'CC EAL6+' },
                    { label: 'Battery', value: '10+ year standby' },
                  ].map((spec) => (
                    <div key={spec.label} className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <p className="text-xs text-gray-500">{spec.label}</p>
                      <p className="text-sm text-white font-medium">{spec.value}</p>
                    </div>
                  ))}
                </div>

                <div className="text-center">
                  <p className="text-lg font-bold text-purple-300">Your VAULT Balance: {vaultBalance}</p>
                  {vaultBalance > 0 && (
                    <button
                      onClick={() => setActiveTab('redeem')}
                      className="mt-3 px-6 py-2.5 rounded-xl text-white font-medium"
                      style={{ background: 'linear-gradient(135deg, #6c5ce7, #a882ff)' }}
                    >
                      Redeem Physical Device
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Redeem Tab */}
            {activeTab === 'redeem' && (
              <div className="space-y-4">
                {vaultBalance < 1 ? (
                  <div className="text-center py-8">
                    <AlertTriangle className="w-12 h-12 text-yellow-400 mx-auto mb-3" />
                    <p className="text-white font-medium">No VAULT Tokens</p>
                    <p className="text-gray-400 text-sm mt-1">You need at least 1 VAULT token to redeem a physical device.</p>
                  </div>
                ) : (
                  <>
                    <h3 className="text-lg font-semibold text-white">Shipping Information</h3>

                    <div className="space-y-3">
                      <input
                        value={shippingName}
                        onChange={(e) => setShippingName(e.target.value)}
                        placeholder="Full Name *"
                        className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                      />
                      <input
                        value={shippingAddress}
                        onChange={(e) => setShippingAddress(e.target.value)}
                        placeholder="Street Address *"
                        className="w-full p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                      />
                      <div className="grid grid-cols-2 gap-3">
                        <input
                          value={city}
                          onChange={(e) => setCity(e.target.value)}
                          placeholder="City *"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                        <input
                          value={stateProvince}
                          onChange={(e) => setStateProvince(e.target.value)}
                          placeholder="State/Province"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <input
                          value={zip}
                          onChange={(e) => setZip(e.target.value)}
                          placeholder="ZIP / Postal Code"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                        <input
                          value={country}
                          onChange={(e) => setCountry(e.target.value)}
                          placeholder="Country *"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <input
                          value={phone}
                          onChange={(e) => setPhone(e.target.value)}
                          placeholder="Phone"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                        <input
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          placeholder="Email *"
                          type="email"
                          className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-purple-500/50 outline-none"
                        />
                      </div>
                    </div>

                    {/* Color Variant Selector */}
                    <div>
                      <p className="text-sm text-gray-400 mb-2">Color Variant</p>
                      <div className="grid grid-cols-3 gap-2">
                        {COLOR_VARIANTS.map((v) => (
                          <button
                            key={v.name}
                            onClick={() => setColorVariant(v.name)}
                            className={`p-3 rounded-xl text-center transition-all ${
                              colorVariant === v.name
                                ? 'ring-2 ring-purple-400 ring-offset-2 ring-offset-transparent'
                                : 'hover:bg-white/5'
                            }`}
                            style={{
                              background: `linear-gradient(135deg, ${v.color}, ${v.accent})`,
                              border: colorVariant === v.name ? '2px solid rgba(168, 130, 255, 0.6)' : '1px solid rgba(255,255,255,0.1)',
                            }}
                          >
                            <p className="text-xs font-medium text-white drop-shadow-lg">{v.name}</p>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Burn Confirmation */}
                    <label className="flex items-start gap-3 p-3 rounded-xl cursor-pointer" style={{ background: 'rgba(234, 88, 12, 0.1)', border: '1px solid rgba(234, 88, 12, 0.2)' }}>
                      <input
                        type="checkbox"
                        checked={confirmBurn}
                        onChange={(e) => setConfirmBurn(e.target.checked)}
                        className="mt-0.5 accent-purple-500"
                      />
                      <span className="text-sm text-orange-300">
                        I understand that 1 VAULT token will be permanently burned to redeem a physical Quillon Vault hardware wallet.
                        This action cannot be undone.
                      </span>
                    </label>

                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleRedeem}
                      disabled={submitting || !confirmBurn}
                      className="w-full py-3.5 rounded-xl font-bold text-white disabled:opacity-50 flex items-center justify-center gap-2"
                      style={{ background: confirmBurn ? 'linear-gradient(135deg, #6c5ce7, #a882ff)' : 'rgba(100,100,100,0.3)' }}
                    >
                      {submitting ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <>
                          <Package className="w-5 h-5" />
                          Burn 1 VAULT & Place Order
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
                    <Loader2 className="w-8 h-8 text-purple-400 animate-spin mx-auto" />
                  </div>
                ) : redemptions.length === 0 ? (
                  <div className="text-center py-8">
                    <Package className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">No redemption orders yet.</p>
                    {vaultBalance > 0 && (
                      <button
                        onClick={() => setActiveTab('redeem')}
                        className="mt-3 text-purple-400 text-sm hover:text-purple-300"
                      >
                        Redeem a device →
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {redemptions.map((r) => {
                      const StatusIcon = STATUS_ICONS[r.status] || Package;
                      return (
                        <div key={r.redemption_id} className="p-4 rounded-xl" style={{ background: 'rgba(168, 130, 255, 0.05)', border: '1px solid rgba(168, 130, 255, 0.12)' }}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-mono text-purple-300">{r.redemption_id}</span>
                            <span className={`text-sm font-medium flex items-center gap-1.5 ${STATUS_COLORS[r.status] || 'text-gray-400'}`}>
                              <StatusIcon className="w-4 h-4" />
                              {r.status.charAt(0).toUpperCase() + r.status.slice(1)}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                            <p className="text-gray-400">Color: <span className="text-white">{r.color_variant}</span></p>
                            <p className="text-gray-400">Qty: <span className="text-white">{r.quantity}</span></p>
                            <p className="text-gray-400">Ship to: <span className="text-white">{r.shipping_name}</span></p>
                            <p className="text-gray-400">
                              Date: <span className="text-white">{new Date(r.created_at * 1000).toLocaleDateString()}</span>
                            </p>
                          </div>
                          {r.tracking_number && (
                            <p className="text-sm text-blue-400 mt-2">
                              <Truck className="w-3.5 h-3.5 inline mr-1" />
                              Tracking: {r.tracking_number}
                            </p>
                          )}
                          {r.serial_number && (
                            <p className="text-sm text-green-400">
                              <Shield className="w-3.5 h-3.5 inline mr-1" />
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
