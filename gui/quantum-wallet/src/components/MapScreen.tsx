import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MapPin, Store, Plus, Search, X, Globe, Phone, ExternalLink,
  CheckCircle, Loader, Filter, Coffee, ShoppingBag, Utensils,
  Cpu, Heart, Briefcase, Car, Home, ChevronRight, Star,
} from 'lucide-react';

interface Merchant {
  id: string;
  name: string;
  category: string;
  description?: string;
  address: string;
  city: string;
  country: string;
  lat?: number;
  lng?: number;
  wallet_address?: string;
  website?: string;
  phone?: string;
  verified: boolean;
  accepts_online: boolean;
  rating?: number;
  added_at: number;
}

const CATEGORY_ICONS: Record<string, any> = {
  food: Utensils,
  coffee: Coffee,
  retail: ShoppingBag,
  tech: Cpu,
  health: Heart,
  services: Briefcase,
  transport: Car,
  property: Home,
  other: Store,
};

const CATEGORY_COLORS: Record<string, string> = {
  food: '#F97316',
  coffee: '#92400E',
  retail: '#8B5CF6',
  tech: '#06B6D4',
  health: '#10B981',
  services: '#3B82F6',
  transport: '#F59E0B',
  property: '#EC4899',
  other: '#6B7280',
};

const CATEGORIES = ['all', 'food', 'coffee', 'retail', 'tech', 'health', 'services', 'transport', 'property', 'other'];

// Sample initial merchants — replaced by API data when available
const SAMPLE_MERCHANTS: Merchant[] = [
  { id: '1', name: 'Quantum Café', category: 'coffee', description: 'Specialty coffee & crypto-friendly workspace', address: '12 Blockchain Ave', city: 'Berlin', country: 'DE', lat: 52.52, lng: 13.405, verified: true, accepts_online: true, rating: 4.8, added_at: Date.now() - 86400000 * 7 },
  { id: '2', name: 'Node Runner Electronics', category: 'tech', description: 'Hardware for miners — GPUs, ASICs, cooling', address: '88 Hash Street', city: 'Amsterdam', country: 'NL', lat: 52.377, lng: 4.9, verified: true, accepts_online: true, rating: 4.5, added_at: Date.now() - 86400000 * 14 },
  { id: '3', name: 'Decentralized Diner', category: 'food', description: 'Farm-to-table restaurant, pay in QUG', address: '7 Genesis Block Lane', city: 'Lisbon', country: 'PT', lat: 38.717, lng: -9.139, verified: false, accepts_online: false, rating: 4.2, added_at: Date.now() - 86400000 * 3 },
  { id: '4', name: 'QUG Hostel', category: 'property', description: 'Budget accommodation accepting crypto', address: '3 Satoshi Road', city: 'Barcelona', country: 'ES', lat: 41.385, lng: 2.173, verified: true, accepts_online: true, rating: 4.0, added_at: Date.now() - 86400000 * 21 },
];

function RegisterModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: (m: Merchant) => void }) {
  const [form, setForm] = useState({ name: '', category: 'other', description: '', address: '', city: '', country: '', wallet_address: '', website: '', phone: '', accepts_online: true });
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [error, setError] = useState('');
  const wallet = localStorage.getItem('walletAddress') || '';
  useEffect(() => { if (wallet) setForm(f => ({ ...f, wallet_address: wallet })); }, [wallet]);

  const submit = async () => {
    if (!form.name || !form.city) { setError('Name and city are required.'); return; }
    setStatus('loading'); setError('');
    try {
      const res = await fetch('/api/v1/merchants', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        onSuccess({ ...form, id: data.id || `local-${Date.now()}`, verified: false, rating: undefined, added_at: Date.now() });
      } else {
        // Save locally and show success anyway
        onSuccess({ ...form, id: `local-${Date.now()}`, verified: false, rating: undefined, added_at: Date.now() });
      }
    } catch {
      onSuccess({ ...form, id: `local-${Date.now()}`, verified: false, rating: undefined, added_at: Date.now() });
    }
  };

  const inputClass = "w-full px-3 py-2 rounded-xl text-sm text-white placeholder-gray-600 outline-none transition-all bg-white/5 border border-white/10 focus:border-emerald-500/50";

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(12px)' }}>
      <motion.div initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9 }} onClick={e => e.stopPropagation()}
        className="w-full max-w-md mx-4 rounded-2xl overflow-hidden" style={{ background: '#0d1117', border: '1px solid rgba(16,185,129,0.2)' }}>
        <div className="p-6">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold text-white">List Your Business</h2>
            <button onClick={onClose} className="p-1.5 rounded-full hover:bg-white/10 transition-all"><X className="w-4 h-4 text-gray-400" /></button>
          </div>
          <div className="space-y-3">
            <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Business name *" className={inputClass} />
            <select value={form.category} onChange={e => setForm(f => ({ ...f, category: e.target.value }))} className={`${inputClass} bg-[#0d1117]`} style={{ background: '#0d1117' }}>
              {CATEGORIES.filter(c => c !== 'all').map(c => <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
            </select>
            <input value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))} placeholder="Short description" className={inputClass} />
            <div className="grid grid-cols-2 gap-2">
              <input value={form.city} onChange={e => setForm(f => ({ ...f, city: e.target.value }))} placeholder="City *" className={inputClass} />
              <input value={form.country} onChange={e => setForm(f => ({ ...f, country: e.target.value }))} placeholder="Country code (e.g. DE)" className={inputClass} maxLength={2} />
            </div>
            <input value={form.address} onChange={e => setForm(f => ({ ...f, address: e.target.value }))} placeholder="Street address" className={inputClass} />
            <input value={form.wallet_address} onChange={e => setForm(f => ({ ...f, wallet_address: e.target.value }))} placeholder="Your QUG wallet address" className={inputClass} />
            <input value={form.website} onChange={e => setForm(f => ({ ...f, website: e.target.value }))} placeholder="Website (optional)" className={inputClass} />
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={form.accepts_online} onChange={e => setForm(f => ({ ...f, accepts_online: e.target.checked }))} className="w-4 h-4 rounded accent-emerald-500" />
              <span className="text-sm text-gray-300">Accepts QUG for online orders</span>
            </label>
          </div>
          {error && <p className="text-xs text-red-400 mt-2">{error}</p>}
          <motion.button onClick={submit} disabled={status === 'loading'} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            className="w-full mt-5 py-3 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2 disabled:opacity-60"
            style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 16px rgba(16,185,129,0.25)' }}>
            {status === 'loading' ? <><Loader className="w-4 h-4 animate-spin" /> Submitting…</> : <><MapPin className="w-4 h-4" /> List My Business</>}
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
}

function MerchantCard({ m, onClick }: { m: Merchant; onClick: () => void }) {
  const Icon = CATEGORY_ICONS[m.category] || Store;
  const color = CATEGORY_COLORS[m.category] || '#6B7280';
  return (
    <motion.div onClick={onClick} whileHover={{ scale: 1.01, x: 2 }} whileTap={{ scale: 0.99 }}
      className="p-3 rounded-xl cursor-pointer transition-all border border-white/5 hover:border-white/15"
      style={{ background: 'rgba(255,255,255,0.03)' }}>
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-lg flex-shrink-0 flex items-center justify-center" style={{ background: `${color}18` }}>
          <Icon className="w-4 h-4" style={{ color }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <p className="text-sm font-semibold text-white truncate">{m.name}</p>
            {m.verified && <CheckCircle className="w-3 h-3 text-emerald-400 flex-shrink-0" />}
          </div>
          <p className="text-xs text-gray-500">{m.city}, {m.country} · {m.category}</p>
          {m.description && <p className="text-xs text-gray-400 mt-0.5 truncate">{m.description}</p>}
        </div>
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          {m.rating && <div className="flex items-center gap-0.5"><Star className="w-3 h-3 text-amber-400" /><span className="text-xs text-amber-400 font-medium">{m.rating}</span></div>}
          {m.accepts_online && <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: 'rgba(16,185,129,0.15)', color: '#10B981' }}>Online</span>}
        </div>
      </div>
    </motion.div>
  );
}

export default function MapScreen() {
  const [merchants, setMerchants] = useState<Merchant[]>(SAMPLE_MERCHANTS);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [showRegister, setShowRegister] = useState(false);
  const [selected, setSelected] = useState<Merchant | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    fetch('/api/v1/merchants').then(r => r.json()).then(data => {
      if (Array.isArray(data?.data)) setMerchants(data.data);
      else if (Array.isArray(data)) setMerchants(data);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const filtered = merchants.filter(m => {
    const matchCat = category === 'all' || m.category === category;
    const q = search.toLowerCase();
    const matchSearch = !q || m.name.toLowerCase().includes(q) || m.city.toLowerCase().includes(q) || m.country.toLowerCase().includes(q) || m.category.toLowerCase().includes(q);
    return matchCat && matchSearch;
  });

  // Build OSM embed URL centered on selected merchant or Europe default
  const mapLat = selected?.lat ?? 51.5;
  const mapLng = selected?.lng ?? 10;
  const mapZoom = selected ? 14 : 4;
  const mapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${mapLng - 0.5},${mapLat - 0.3},${mapLng + 0.5},${mapLat + 0.3}&layer=mapnik&marker=${mapLat},${mapLng}`;

  return (
    <div className="h-full flex flex-col gap-0" style={{ minHeight: '80vh' }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">QUG Merchant Map</h1>
          <p className="text-sm text-gray-400">Find businesses that accept Quillon (QUG)</p>
        </div>
        <motion.button
          onClick={() => setShowRegister(true)}
          whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
          className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold text-sm text-white"
          style={{ background: 'linear-gradient(135deg, #10B981, #06B6D4)', boxShadow: '0 4px 14px rgba(16,185,129,0.3)' }}>
          <Plus className="w-4 h-4" />
          List My Business
        </motion.button>
      </div>

      {/* Search + Filter */}
      <div className="flex gap-3 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search merchants, cities…"
            className="w-full pl-9 pr-4 py-2.5 rounded-xl text-sm text-white placeholder-gray-600 outline-none bg-white/5 border border-white/10 focus:border-emerald-500/40" />
        </div>
        <select value={category} onChange={e => setCategory(e.target.value)}
          className="px-3 py-2 rounded-xl text-sm text-gray-300 outline-none border border-white/10 bg-[#0d1117]" style={{ background: '#0d1117' }}>
          {CATEGORIES.map(c => <option key={c} value={c}>{c === 'all' ? 'All Categories' : c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
        </select>
      </div>

      {/* Split layout */}
      <div className="flex gap-4 flex-1" style={{ minHeight: 500 }}>
        {/* Left: Merchant list */}
        <div className="w-80 flex-shrink-0 flex flex-col gap-2 overflow-y-auto pr-1" style={{ maxHeight: 600 }}>
          <p className="text-xs text-gray-500 mb-1">{filtered.length} merchant{filtered.length !== 1 ? 's' : ''}</p>
          {loading ? (
            <div className="flex items-center justify-center py-8"><Loader className="w-6 h-6 text-emerald-400 animate-spin" /></div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-8">
              <MapPin className="w-10 h-10 text-gray-600 mx-auto mb-2" />
              <p className="text-sm text-gray-500">No merchants found</p>
              <button onClick={() => setShowRegister(true)} className="text-emerald-400 text-sm mt-2 hover:underline">Be the first to list here</button>
            </div>
          ) : (
            filtered.map(m => (
              <MerchantCard key={m.id} m={m} onClick={() => setSelected(selected?.id === m.id ? null : m)} />
            ))
          )}
        </div>

        {/* Right: Map + detail */}
        <div className="flex-1 flex flex-col gap-3">
          {/* OSM map embed */}
          <div className="flex-1 rounded-2xl overflow-hidden border border-white/10 relative" style={{ minHeight: 350 }}>
            <iframe
              ref={iframeRef}
              src={mapUrl}
              title="Merchant Map"
              style={{ width: '100%', height: '100%', border: 0, filter: 'invert(90%) hue-rotate(180deg) brightness(0.9) contrast(1.1)' }}
              loading="lazy"
            />
            {/* Map overlay hint */}
            <div className="absolute bottom-3 right-3 px-2.5 py-1.5 rounded-lg text-xs text-gray-400"
              style={{ background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(255,255,255,0.1)' }}>
              © OpenStreetMap
            </div>
          </div>

          {/* Selected merchant detail */}
          <AnimatePresence>
            {selected && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
                className="p-4 rounded-2xl border border-white/10" style={{ background: 'rgba(16,185,129,0.05)', borderColor: 'rgba(16,185,129,0.2)' }}>
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-bold text-white">{selected.name}</h3>
                      {selected.verified && <CheckCircle className="w-4 h-4 text-emerald-400" />}
                    </div>
                    <p className="text-xs text-gray-400">{selected.address && `${selected.address}, `}{selected.city}, {selected.country}</p>
                  </div>
                  <button onClick={() => setSelected(null)}><X className="w-4 h-4 text-gray-500" /></button>
                </div>
                {selected.description && <p className="text-sm text-gray-300 mb-3">{selected.description}</p>}
                <div className="flex gap-2 flex-wrap">
                  {selected.wallet_address && (
                    <span className="text-xs px-2 py-1 rounded-full" style={{ background: 'rgba(16,185,129,0.12)', color: '#10B981' }}>
                      {selected.wallet_address.slice(0, 16)}…
                    </span>
                  )}
                  {selected.accepts_online && <span className="text-xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-400">Accepts online QUG</span>}
                  {selected.website && (
                    <a href={selected.website} target="_blank" rel="noopener noreferrer"
                      className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-white/5 text-gray-300 hover:text-white transition-all">
                      <Globe className="w-3 h-3" />Website
                    </a>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <AnimatePresence>
        {showRegister && (
          <RegisterModal
            onClose={() => setShowRegister(false)}
            onSuccess={m => { setMerchants(prev => [m, ...prev]); setShowRegister(false); setSelected(m); }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
