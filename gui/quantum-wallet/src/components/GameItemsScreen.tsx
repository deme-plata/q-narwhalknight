import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Package, Search, Filter, ChevronDown, Star, Crosshair,
  ShoppingCart, ArrowUpDown, Sparkles, Trophy, X, ExternalLink,
  Sword, Shield, Gem, Lock, Unlock, Tag, TrendingUp, Eye
} from 'lucide-react';

// ─── Types ──────────────────────────────────────────────────────────────────

interface GameItem {
  id: string;
  name: string;
  weapon: string;
  collection: string;
  grade: string;
  wear: string;
  float_value: number;
  item_type: string;
  owner_wallet: string;
  stattrak: boolean;
  trade_count: number;
  created_at: number;
  listed_price: number | null;
  case_open_seed: string | null;
}

interface MarketplaceListing {
  listing_id: string;
  item_id: string;
  item_name: string;
  item_weapon: string;
  grade: string;
  wear: string;
  float_value: number;
  stattrak: boolean;
  seller_wallet: string;
  price_qug: number;
  listed_at: number;
  status: string;
}

interface CaseTemplate {
  id: string;
  name: string;
  description: string;
  image_theme: string;
  items_by_grade: Record<string, { name: string; weapon: string; item_type: string }[]>;
  key_price_qug: number;
  has_contraband_pool: boolean;
}

interface CollectionProgress {
  collection: {
    id: string;
    name: string;
    description: string;
    required_items: string[];
    reward_description: string;
    reward_qug: number;
  };
  owned: number;
  total: number;
  complete: boolean;
}

interface MarketStats {
  total_items: number;
  active_listings: number;
  total_sold: number;
  total_volume_qug: number;
  floor_prices: Record<string, number>;
  grade_distribution: Record<string, number>;
}

interface CaseOpenResult {
  item: GameItem;
  grade_info: { name: string; color: string; drop_rate: string };
  wear_info: { name: string; short: string };
  provably_fair: { block_hash: string; wallet: string; nonce: number; seed: string; algorithm: string };
  key_cost_qug: number;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const GRADE_COLORS: Record<string, string> = {
  consumer_grade: '#b0c3d9',
  industrial_grade: '#5e98d9',
  mil_spec: '#4b69ff',
  restricted: '#8847ff',
  classified: '#d32ce6',
  covert: '#eb4b4b',
  contraband: '#e4ae39',
};

const GRADE_NAMES: Record<string, string> = {
  consumer_grade: 'Consumer Grade',
  industrial_grade: 'Industrial Grade',
  mil_spec: 'Mil-Spec',
  restricted: 'Restricted',
  classified: 'Classified',
  covert: 'Covert',
  contraband: 'Contraband',
};

const WEAR_NAMES: Record<string, string> = {
  factory_new: 'Factory New',
  minimal_wear: 'Minimal Wear',
  field_tested: 'Field-Tested',
  well_worn: 'Well-Worn',
  battle_scared: 'Battle-Scarred',
};

const CASE_THEMES: Record<string, string> = {
  quantum: 'from-violet-600 to-cyan-500',
  genesis: 'from-amber-600 to-yellow-400',
  narwhal: 'from-blue-700 to-teal-400',
  covert: 'from-gray-800 to-red-600',
  founders: 'from-amber-500 to-orange-600',
};

const API_BASE = '/api/v1/game-items';

// ─── Utility ────────────────────────────────────────────────────────────────

function getAuthHeaders(): Record<string, string> {
  const authToken = localStorage.getItem('walletAuthToken');
  if (authToken) return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` };
  return { 'Content-Type': 'application/json' };
}

function gradeColor(grade: string): string {
  return GRADE_COLORS[grade] || '#b0c3d9';
}

function gradeName(grade: string): string {
  return GRADE_NAMES[grade] || grade;
}

function wearName(wear: string): string {
  return WEAR_NAMES[wear] || wear;
}

function shortWallet(w: string): string {
  if (!w || w.length < 12) return w;
  return `${w.slice(0, 6)}...${w.slice(-4)}`;
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000) - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ─── Components ─────────────────────────────────────────────────────────────

function FloatBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  const color = value < 0.07 ? '#22c55e' : value < 0.15 ? '#86efac' : value < 0.38 ? '#facc15' : value < 0.45 ? '#f97316' : '#ef4444';
  return (
    <div className="w-full h-1.5 rounded-full bg-slate-700 overflow-hidden" title={`Float: ${value.toFixed(6)}`}>
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

function GradeBadge({ grade }: { grade: string }) {
  return (
    <span
      className="text-xs font-bold px-2 py-0.5 rounded-full"
      style={{ backgroundColor: `${gradeColor(grade)}22`, color: gradeColor(grade), border: `1px solid ${gradeColor(grade)}44` }}
    >
      {gradeName(grade)}
    </span>
  );
}

function ItemCard({ item, onAction, actionLabel }: { item: GameItem | MarketplaceListing; onAction?: () => void; actionLabel?: string }) {
  const isListing = 'listing_id' in item;
  const name = isListing ? (item as MarketplaceListing).item_name : (item as GameItem).name;
  const weapon = isListing ? (item as MarketplaceListing).item_weapon : (item as GameItem).weapon;
  const grade = isListing ? (item as MarketplaceListing).grade : (item as GameItem).grade;
  const wear = isListing ? (item as MarketplaceListing).wear : (item as GameItem).wear;
  const floatVal = isListing ? (item as MarketplaceListing).float_value : (item as GameItem).float_value;
  const stattrak = isListing ? (item as MarketplaceListing).stattrak : (item as GameItem).stattrak;
  const price = isListing ? (item as MarketplaceListing).price_qug : (item as GameItem).listed_price;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl overflow-hidden cursor-pointer group"
      style={{ background: 'linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95))', border: `1px solid ${gradeColor(grade)}33` }}
    >
      {/* Item visual area */}
      <div className="relative h-36 flex items-center justify-center" style={{ background: `linear-gradient(135deg, ${gradeColor(grade)}11, ${gradeColor(grade)}22)` }}>
        <Crosshair className="w-16 h-16 opacity-20" style={{ color: gradeColor(grade) }} />
        {stattrak && (
          <div className="absolute top-2 left-2 bg-orange-500/20 text-orange-400 text-[10px] font-bold px-1.5 py-0.5 rounded">
            StatTrak™
          </div>
        )}
        <div className="absolute top-2 right-2">
          <GradeBadge grade={grade} />
        </div>
      </div>

      <div className="p-3 space-y-2">
        <div>
          <p className="text-sm font-semibold text-slate-200 truncate">{weapon} | {name}</p>
          <p className="text-xs text-slate-400">{wearName(wear)}</p>
        </div>
        <FloatBar value={floatVal} />
        <div className="flex items-center justify-between">
          {price != null && price > 0 ? (
            <span className="text-amber-400 font-bold text-sm">{price.toFixed(2)} QUG</span>
          ) : (
            <span className="text-slate-500 text-xs">Not listed</span>
          )}
          {onAction && actionLabel && (
            <button
              onClick={(e) => { e.stopPropagation(); onAction(); }}
              className="text-xs px-3 py-1.5 rounded-lg font-medium transition-all"
              style={{ background: `${gradeColor(grade)}22`, color: gradeColor(grade), border: `1px solid ${gradeColor(grade)}44` }}
            >
              {actionLabel}
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ─── Case Roulette Animation ────────────────────────────────────────────────

function CaseRouletteModal({ result, onClose }: { result: CaseOpenResult; onClose: () => void }) {
  const [phase, setPhase] = useState<'spinning' | 'reveal'>('spinning');

  useEffect(() => {
    const timer = setTimeout(() => setPhase('reveal'), 2500);
    return () => clearTimeout(timer);
  }, []);

  const color = gradeColor(result.item.grade);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        onClick={e => e.stopPropagation()}
        className="w-full max-w-lg mx-4 rounded-2xl overflow-hidden"
        style={{ background: 'linear-gradient(135deg, #0f172a, #1e293b)', border: `2px solid ${color}66` }}
      >
        {phase === 'spinning' ? (
          <div className="p-8 flex flex-col items-center gap-4">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            >
              <Sparkles className="w-16 h-16" style={{ color }} />
            </motion.div>
            <p className="text-xl font-bold text-slate-200">Opening Case...</p>
            <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: color }}
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 2.5, ease: 'easeInOut' }}
              />
            </div>
          </div>
        ) : (
          <div className="p-8 flex flex-col items-center gap-4">
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: 'spring', stiffness: 200, damping: 15 }}
            >
              <div
                className="w-28 h-28 rounded-2xl flex items-center justify-center relative"
                style={{ background: `linear-gradient(135deg, ${color}33, ${color}11)`, border: `2px solid ${color}` }}
              >
                <Crosshair className="w-14 h-14" style={{ color }} />
                {result.item.stattrak && (
                  <div className="absolute -top-2 -right-2 bg-orange-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full">
                    ST
                  </div>
                )}
              </div>
            </motion.div>

            <div className="text-center space-y-1">
              <GradeBadge grade={result.item.grade} />
              <p className="text-xl font-bold text-slate-200 mt-2">{result.item.weapon} | {result.item.name}</p>
              <p className="text-sm text-slate-400">{result.wear_info.name}</p>
              <FloatBar value={result.item.float_value} />
              <p className="text-xs text-slate-500">Float: {result.item.float_value.toFixed(6)}</p>
            </div>

            {result.item.stattrak && (
              <div className="flex items-center gap-2 text-orange-400 text-sm">
                <Star className="w-4 h-4" />
                <span>StatTrak™ Confirmed</span>
              </div>
            )}

            <details className="w-full text-xs text-slate-500">
              <summary className="cursor-pointer hover:text-slate-300 transition-colors flex items-center gap-1">
                <Eye className="w-3 h-3" /> Provably Fair Verification
              </summary>
              <div className="mt-2 p-3 bg-slate-800/50 rounded-lg space-y-1 font-mono break-all">
                <p>Block Hash: {result.provably_fair.block_hash}</p>
                <p>Nonce: {result.provably_fair.nonce}</p>
                <p>Seed: {result.provably_fair.seed}</p>
                <p className="text-slate-400">{result.provably_fair.algorithm}</p>
              </div>
            </details>

            <button
              onClick={onClose}
              className="w-full py-3 rounded-xl font-semibold text-white transition-all hover:brightness-110"
              style={{ background: `linear-gradient(135deg, ${color}, ${color}cc)` }}
            >
              Awesome!
            </button>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

// ─── Main Screen ────────────────────────────────────────────────────────────

type Tab = 'marketplace' | 'cases' | 'inventory' | 'collections';

export default function GameItemsScreen() {
  const [tab, setTab] = useState<Tab>('marketplace');
  const [listings, setListings] = useState<MarketplaceListing[]>([]);
  const [cases, setCases] = useState<CaseTemplate[]>([]);
  const [inventory, setInventory] = useState<GameItem[]>([]);
  const [collections, setCollections] = useState<CollectionProgress[]>([]);
  const [stats, setStats] = useState<MarketStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [gradeFilter, setGradeFilter] = useState('');
  const [sortBy, setSortBy] = useState('newest');

  // Case opening state
  const [openingCase, setOpeningCase] = useState<string | null>(null);
  const [caseResult, setCaseResult] = useState<CaseOpenResult | null>(null);

  // List for sale modal
  const [listModal, setListModal] = useState<GameItem | null>(null);
  const [listPrice, setListPrice] = useState('');

  // Trade-up state
  const [tradeUpItems, setTradeUpItems] = useState<string[]>([]);
  const [tradeUpResult, setTradeUpResult] = useState<GameItem | null>(null);

  const walletAddress = localStorage.getItem('walletAddress') || '';

  const fetchListings = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (search) params.set('search', search);
      if (gradeFilter) params.set('grade', gradeFilter);
      params.set('sort', sortBy);
      const res = await fetch(`${API_BASE}?${params}`);
      const data = await res.json();
      if (data.success) {
        setListings(data.data.listings || []);
      }
    } catch (e) {
      console.error('Failed to fetch listings:', e);
    }
    setLoading(false);
  }, [search, gradeFilter, sortBy]);

  const fetchCases = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/cases`);
      const data = await res.json();
      if (data.success) setCases(data.data.cases || []);
    } catch (e) {
      console.error('Failed to fetch cases:', e);
    }
  }, []);

  const fetchInventory = useCallback(async () => {
    if (!walletAddress) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/inventory/${walletAddress}`);
      const data = await res.json();
      if (data.success) setInventory(data.data.items || []);
    } catch (e) {
      console.error('Failed to fetch inventory:', e);
    }
    setLoading(false);
  }, [walletAddress]);

  const fetchCollections = useCallback(async () => {
    try {
      const params = walletAddress ? `?wallet=${walletAddress}` : '';
      const res = await fetch(`${API_BASE}/collections${params}`);
      const data = await res.json();
      if (data.success) setCollections(data.data.collections || []);
    } catch (e) {
      console.error('Failed to fetch collections:', e);
    }
  }, [walletAddress]);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      const data = await res.json();
      if (data.success) setStats(data.data);
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    if (tab === 'marketplace') fetchListings();
    else if (tab === 'cases') fetchCases();
    else if (tab === 'inventory') fetchInventory();
    else if (tab === 'collections') fetchCollections();
  }, [tab, fetchListings, fetchCases, fetchInventory, fetchCollections, fetchStats]);

  // ─── Actions ────────────────────────────────────────────────────────────

  const handleOpenCase = async (caseId: string) => {
    setOpeningCase(caseId);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/open-case`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ case_id: caseId }),
      });
      const data = await res.json();
      if (data.success) {
        setCaseResult(data.data);
      } else {
        setError(data.error || 'Failed to open case');
        setOpeningCase(null);
      }
    } catch (e) {
      setError('Network error');
      setOpeningCase(null);
    }
  };

  const handleBuy = async (listingId: string) => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/buy`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ listing_id: listingId }),
      });
      const data = await res.json();
      if (data.success) {
        fetchListings();
        fetchInventory();
        fetchStats();
      } else {
        setError(data.error || 'Purchase failed');
      }
    } catch (e) {
      setError('Network error');
    }
  };

  const handleListForSale = async () => {
    if (!listModal || !listPrice) return;
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/list`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ item_id: listModal.id, price_qug: parseFloat(listPrice) }),
      });
      const data = await res.json();
      if (data.success) {
        setListModal(null);
        setListPrice('');
        fetchInventory();
        fetchListings();
        fetchStats();
      } else {
        setError(data.error || 'Listing failed');
      }
    } catch (e) {
      setError('Network error');
    }
  };

  const handleTradeUp = async () => {
    if (tradeUpItems.length !== 10) return;
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/trade-up`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ item_ids: tradeUpItems }),
      });
      const data = await res.json();
      if (data.success) {
        setTradeUpResult(data.data.result_item);
        setTradeUpItems([]);
        fetchInventory();
      } else {
        setError(data.error || 'Trade-up failed');
      }
    } catch (e) {
      setError('Network error');
    }
  };

  const toggleTradeUpItem = (itemId: string) => {
    setTradeUpItems(prev =>
      prev.includes(itemId)
        ? prev.filter(id => id !== itemId)
        : prev.length < 10
          ? [...prev, itemId]
          : prev
    );
  };

  // ─── Tab Buttons ────────────────────────────────────────────────────────

  const tabs: { id: Tab; label: string; icon: typeof Package }[] = [
    { id: 'marketplace', label: 'Marketplace', icon: ShoppingCart },
    { id: 'cases', label: 'Cases', icon: Package },
    { id: 'inventory', label: 'Inventory', icon: Sword },
    { id: 'collections', label: 'Collections', icon: Trophy },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent flex items-center gap-2">
            <Crosshair className="w-7 h-7 text-amber-400" />
            Game Items
          </h1>
          <p className="text-sm text-slate-400 mt-1">CS:GO2-style blockchain collectibles with provably fair drops</p>
        </div>
        {stats && (
          <div className="flex gap-4 text-xs">
            <div className="text-center">
              <p className="text-slate-400">Items</p>
              <p className="text-amber-400 font-bold text-lg">{stats.total_items}</p>
            </div>
            <div className="text-center">
              <p className="text-slate-400">Listed</p>
              <p className="text-green-400 font-bold text-lg">{stats.active_listings}</p>
            </div>
            <div className="text-center">
              <p className="text-slate-400">Volume</p>
              <p className="text-cyan-400 font-bold text-lg">{stats.total_volume_qug.toFixed(1)} QUG</p>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
              tab === t.id
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:text-slate-200'
            }`}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 flex items-center justify-between"
          >
            <span className="text-red-400 text-sm">{error}</span>
            <button onClick={() => setError(null)}><X className="w-4 h-4 text-red-400" /></button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ MARKETPLACE TAB ═══ */}
      {tab === 'marketplace' && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                placeholder="Search items..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:border-amber-500/50"
              />
            </div>
            <select
              value={gradeFilter}
              onChange={e => setGradeFilter(e.target.value)}
              className="px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-200 text-sm focus:outline-none"
            >
              <option value="">All Grades</option>
              {Object.entries(GRADE_NAMES).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              className="px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-200 text-sm focus:outline-none"
            >
              <option value="newest">Newest</option>
              <option value="price_low">Price: Low to High</option>
              <option value="price_high">Price: High to Low</option>
              <option value="grade">Grade: Rarest</option>
            </select>
          </div>

          {/* Listings Grid */}
          {loading ? (
            <div className="text-center py-16 text-slate-500">Loading marketplace...</div>
          ) : listings.length === 0 ? (
            <div className="text-center py-16">
              <ShoppingCart className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">No items listed yet</p>
              <p className="text-slate-500 text-sm mt-1">Open cases to get items, then list them here!</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {listings.map(listing => (
                <ItemCard
                  key={listing.listing_id}
                  item={listing}
                  actionLabel={listing.seller_wallet === walletAddress ? undefined : `Buy ${listing.price_qug.toFixed(2)}`}
                  onAction={listing.seller_wallet === walletAddress ? undefined : () => handleBuy(listing.listing_id)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ═══ CASES TAB ═══ */}
      {tab === 'cases' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {cases.map(c => {
              const theme = CASE_THEMES[c.image_theme] || 'from-slate-700 to-slate-600';
              const allItems = Object.entries(c.items_by_grade).flatMap(([grade, items]) =>
                items.map(i => ({ ...i, grade }))
              );
              return (
                <motion.div
                  key={c.id}
                  whileHover={{ scale: 1.02 }}
                  className="rounded-2xl overflow-hidden"
                  style={{ background: 'linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95))', border: '1px solid rgba(212,175,55,0.15)' }}
                >
                  <div className={`h-40 bg-gradient-to-br ${theme} flex items-center justify-center relative`}>
                    <Package className="w-20 h-20 text-white/30" />
                    {c.has_contraband_pool && (
                      <div className="absolute top-3 right-3 bg-yellow-500/20 text-yellow-400 text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1">
                        <Gem className="w-3 h-3" /> Contraband
                      </div>
                    )}
                  </div>
                  <div className="p-4 space-y-3">
                    <h3 className="text-lg font-bold text-slate-200">{c.name}</h3>
                    <p className="text-xs text-slate-400 line-clamp-2">{c.description}</p>
                    <div className="flex flex-wrap gap-1">
                      {Object.keys(c.items_by_grade).map(grade => (
                        <span
                          key={grade}
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: gradeColor(grade) }}
                          title={`${gradeName(grade)}: ${c.items_by_grade[grade].length} items`}
                        />
                      ))}
                      <span className="text-xs text-slate-500 ml-1">{allItems.length} items</span>
                    </div>
                    <div className="flex items-center justify-between pt-2">
                      <span className="text-amber-400 font-bold">{c.key_price_qug.toFixed(2)} QUG</span>
                      <button
                        onClick={() => handleOpenCase(c.id)}
                        disabled={openingCase === c.id}
                        className="px-4 py-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold text-sm hover:brightness-110 transition-all disabled:opacity-50"
                      >
                        {openingCase === c.id ? 'Opening...' : 'Open Case'}
                      </button>
                    </div>
                  </div>

                  {/* Expandable item list */}
                  <details className="px-4 pb-4">
                    <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-300 transition-colors flex items-center gap-1">
                      <ChevronDown className="w-3 h-3" /> View possible drops
                    </summary>
                    <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
                      {Object.entries(c.items_by_grade)
                        .sort(([a], [b]) => {
                          const order = ['contraband', 'covert', 'classified', 'restricted', 'mil_spec', 'industrial_grade', 'consumer_grade'];
                          return order.indexOf(a) - order.indexOf(b);
                        })
                        .map(([grade, items]) =>
                          items.map((item, idx) => (
                            <div key={`${grade}-${idx}`} className="flex items-center justify-between py-1">
                              <div className="flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: gradeColor(grade) }} />
                                <span className="text-xs text-slate-300">{item.weapon} | {item.name}</span>
                              </div>
                              <span className="text-[10px] text-slate-500">{gradeName(grade)}</span>
                            </div>
                          ))
                        )}
                    </div>
                  </details>
                </motion.div>
              );
            })}
          </div>
          {cases.length === 0 && (
            <div className="text-center py-16 text-slate-500">Loading cases...</div>
          )}
        </div>
      )}

      {/* ═══ INVENTORY TAB ═══ */}
      {tab === 'inventory' && (
        <div className="space-y-4">
          {!walletAddress ? (
            <div className="text-center py-16 text-slate-500">Connect your wallet to view inventory</div>
          ) : (
            <>
              {/* Trade-Up Mode */}
              {tradeUpItems.length > 0 && (
                <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-bold text-purple-400 flex items-center gap-2">
                      <ArrowUpDown className="w-4 h-4" />
                      Trade-Up Contract: {tradeUpItems.length}/10 items selected
                    </h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setTradeUpItems([])}
                        className="text-xs px-3 py-1 rounded-lg bg-slate-700 text-slate-300"
                      >
                        Cancel
                      </button>
                      {tradeUpItems.length === 10 && (
                        <button
                          onClick={handleTradeUp}
                          className="text-xs px-3 py-1 rounded-lg bg-purple-600 text-white font-semibold"
                        >
                          Trade Up!
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <div
                        key={i}
                        className={`h-1.5 flex-1 rounded-full ${i < tradeUpItems.length ? 'bg-purple-500' : 'bg-slate-700'}`}
                      />
                    ))}
                  </div>
                </div>
              )}

              {loading ? (
                <div className="text-center py-16 text-slate-500">Loading inventory...</div>
              ) : inventory.length === 0 ? (
                <div className="text-center py-16">
                  <Sword className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">Your inventory is empty</p>
                  <p className="text-slate-500 text-sm mt-1">Open cases to get your first items!</p>
                  <button
                    onClick={() => setTab('cases')}
                    className="mt-4 px-6 py-2 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 text-sm font-medium"
                  >
                    Browse Cases
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                  {inventory.map(item => (
                    <div key={item.id} className="relative">
                      {tradeUpItems.includes(item.id) && (
                        <div className="absolute -top-1 -right-1 z-10 w-5 h-5 rounded-full bg-purple-500 flex items-center justify-center text-white text-[10px] font-bold">
                          {tradeUpItems.indexOf(item.id) + 1}
                        </div>
                      )}
                      <div onClick={() => tradeUpItems.length > 0 ? toggleTradeUpItem(item.id) : null}>
                        <ItemCard
                          item={item}
                          actionLabel={item.listed_price ? 'Listed' : 'Sell'}
                          onAction={item.listed_price ? undefined : () => setListModal(item)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Trade-up start button */}
              {inventory.length >= 10 && tradeUpItems.length === 0 && (
                <div className="flex justify-center">
                  <button
                    onClick={() => setTradeUpItems([])} // Shows the trade-up UI bar
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/30 text-sm hover:bg-purple-500/20 transition-all"
                  >
                    <ArrowUpDown className="w-4 h-4" />
                    Start Trade-Up Contract (select 10 same-grade items)
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ═══ COLLECTIONS TAB ═══ */}
      {tab === 'collections' && (
        <div className="space-y-4">
          {collections.length === 0 ? (
            <div className="text-center py-16 text-slate-500">Loading collections...</div>
          ) : (
            collections.map(cp => (
              <motion.div
                key={cp.collection.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl p-5"
                style={{
                  background: 'linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95))',
                  border: cp.complete ? '1px solid rgba(34,197,94,0.4)' : '1px solid rgba(100,116,139,0.2)',
                }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                      {cp.complete && <Trophy className="w-5 h-5 text-green-400" />}
                      {cp.collection.name}
                    </h3>
                    <p className="text-sm text-slate-400 mt-1">{cp.collection.description}</p>
                  </div>
                  <div className="text-right">
                    <span className={`text-2xl font-bold ${cp.complete ? 'text-green-400' : 'text-slate-300'}`}>
                      {cp.owned}/{cp.total}
                    </span>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden mb-3">
                  <div
                    className={`h-full rounded-full transition-all ${cp.complete ? 'bg-green-500' : 'bg-amber-500'}`}
                    style={{ width: `${(cp.owned / cp.total) * 100}%` }}
                  />
                </div>

                {/* Required items */}
                <div className="flex flex-wrap gap-2 mb-3">
                  {cp.collection.required_items.map(name => (
                    <span
                      key={name}
                      className={`text-xs px-2 py-1 rounded-lg ${
                        walletAddress
                          ? 'bg-slate-700/50 text-slate-300'
                          : 'bg-slate-800/50 text-slate-500'
                      }`}
                    >
                      {name}
                    </span>
                  ))}
                </div>

                {/* Reward */}
                <div className="flex items-center gap-2 text-sm">
                  <Gem className="w-4 h-4 text-amber-400" />
                  <span className="text-slate-400">Reward:</span>
                  <span className="text-amber-400 font-semibold">{cp.collection.reward_description}</span>
                </div>
              </motion.div>
            ))
          )}
        </div>
      )}

      {/* ═══ MODALS ═══ */}

      {/* List for sale modal */}
      <AnimatePresence>
        {listModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setListModal(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={e => e.stopPropagation()}
              className="w-full max-w-sm mx-4 rounded-2xl p-6 space-y-4"
              style={{ background: 'linear-gradient(135deg, #0f172a, #1e293b)', border: '1px solid rgba(212,175,55,0.3)' }}
            >
              <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                <Tag className="w-5 h-5 text-amber-400" />
                List for Sale
              </h3>
              <div className="text-sm text-slate-300">
                {listModal.weapon} | {listModal.name}
                <GradeBadge grade={listModal.grade} />
              </div>
              <input
                type="number"
                step="0.01"
                min="0.01"
                placeholder="Price in QUG"
                value={listPrice}
                onChange={e => setListPrice(e.target.value)}
                className="w-full px-4 py-3 rounded-xl bg-slate-800 border border-slate-700 text-slate-200 focus:outline-none focus:border-amber-500/50"
              />
              <div className="flex gap-3">
                <button
                  onClick={() => setListModal(null)}
                  className="flex-1 py-2.5 rounded-xl bg-slate-700 text-slate-300 text-sm font-medium"
                >
                  Cancel
                </button>
                <button
                  onClick={handleListForSale}
                  disabled={!listPrice || parseFloat(listPrice) <= 0}
                  className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-white text-sm font-semibold disabled:opacity-50"
                >
                  List for {listPrice || '0'} QUG
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Case opening result */}
      <AnimatePresence>
        {caseResult && (
          <CaseRouletteModal
            result={caseResult}
            onClose={() => {
              setCaseResult(null);
              setOpeningCase(null);
              fetchInventory();
              fetchStats();
            }}
          />
        )}
      </AnimatePresence>

      {/* Trade-up result */}
      <AnimatePresence>
        {tradeUpResult && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
            onClick={() => setTradeUpResult(null)}
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              onClick={e => e.stopPropagation()}
              className="w-full max-w-md mx-4 rounded-2xl p-8 text-center space-y-4"
              style={{
                background: 'linear-gradient(135deg, #0f172a, #1e293b)',
                border: `2px solid ${gradeColor(tradeUpResult.grade)}66`,
              }}
            >
              <ArrowUpDown className="w-12 h-12 mx-auto" style={{ color: gradeColor(tradeUpResult.grade) }} />
              <h3 className="text-xl font-bold text-slate-200">Trade-Up Complete!</h3>
              <GradeBadge grade={tradeUpResult.grade} />
              <p className="text-lg text-slate-200">{tradeUpResult.weapon} | {tradeUpResult.name}</p>
              <p className="text-sm text-slate-400">{wearName(tradeUpResult.wear)}</p>
              <FloatBar value={tradeUpResult.float_value} />
              {tradeUpResult.stattrak && (
                <p className="text-orange-400 text-sm font-bold">StatTrak™</p>
              )}
              <button
                onClick={() => setTradeUpResult(null)}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold"
              >
                Nice!
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
