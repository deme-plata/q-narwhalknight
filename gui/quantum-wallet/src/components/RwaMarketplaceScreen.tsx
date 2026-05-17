import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Building, TrendingUp, Landmark, Gem, Leaf, Palette, FileText, Package,
  Search, Grid3X3, List, Shield, ChevronRight, ExternalLink, X,
  ArrowUpRight, Clock, DollarSign, BarChart3, Loader2, ShoppingCart,
  CheckCircle2, AlertCircle, Filter, SortDesc, Briefcase, Star, Users,
  Crosshair
} from 'lucide-react';
import XListCrowdfundModal from './XListCrowdfundModal';

// ─── Types ───────────────────────────────────────────────────────────────────

interface RwaListing {
  address: string;
  name: string;
  symbol: string;
  contract_type: string;
  category: string;
  description: string;
  deployed_at: number;
  verified: boolean;
  features: Record<string, boolean>;
  total_value_usd: string;
  shares_available: string;
  kyc_required: boolean;
  dividend_enabled: boolean;
  // Campaign-specific (only set for ExchangeListing type)
  campaign_id?: string;
  raised_usd?: number;
  target_usd_num?: number;
  progress_percent?: number;
  contributor_count?: number;
  campaign_status?: string;
}

interface PortfolioHolding {
  address: string;
  name: string;
  symbol: string;
  type: string;
  sharesOwned: number;
  currentValue: number;
  costBasis: number;
  yieldEarned: number;
  deployedAt: number;
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const CATEGORY_MAP: Record<string, { label: string; icon: typeof Building; apiKey: string }> = {
  all:             { label: 'All Assets',         icon: Briefcase,  apiKey: '' },
  real_estate:     { label: 'Real Estate',        icon: Building,   apiKey: 'real_estate' },
  equity:          { label: 'Equity & Shares',    icon: TrendingUp, apiKey: 'equity' },
  fixed_income:    { label: 'Fixed Income',       icon: Landmark,   apiKey: 'fixed_income' },
  commodity:       { label: 'Commodities',        icon: Gem,        apiKey: 'commodity' },
  carbon_credit:   { label: 'Carbon Credits',     icon: Leaf,       apiKey: 'carbon_credit' },
  art_collectible: { label: 'Art & Collectibles', icon: Palette,    apiKey: 'art_collectible' },
  ip_revenue:      { label: 'IP & Royalties',     icon: FileText,   apiKey: 'ip_revenue' },
  physical_goods:      { label: 'Physical Goods',     icon: Package,    apiKey: 'physical_goods' },
  exchange_listing:    { label: 'Exchange Listings',  icon: Star,       apiKey: 'exchange_listing' },
  game_items:          { label: 'Game Items',         icon: Crosshair,  apiKey: 'game_items' },
};

const SORT_OPTIONS: { value: string; label: string }[] = [
  { value: 'newest',     label: 'Newest First' },
  { value: 'value_high', label: 'Highest Value' },
  { value: 'value_low',  label: 'Lowest Value' },
  { value: 'yield',      label: 'Dividend Enabled' },
];

// ─── Utility Functions ───────────────────────────────────────────────────────

const formatCurrency = (value: string | number): string => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num) || num === 0) return '$0';
  if (num >= 1_000_000_000) return `$${(num / 1_000_000_000).toFixed(2)}B`;
  if (num >= 1_000_000) return `$${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `$${(num / 1_000).toFixed(1)}K`;
  return `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const formatNumber = (value: string | number): string => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num)) return '0';
  return num.toLocaleString();
};

const formatTimestamp = (ts: number): string => {
  if (!ts) return 'Unknown';
  const date = new Date(ts * 1000);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const timeAgo = (ts: number): string => {
  if (!ts) return '';
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 2592000) return `${Math.floor(seconds / 86400)}d ago`;
  return formatTimestamp(ts);
};

const pricePerShare = (totalValue: string, shares: string): number => {
  const tv = parseFloat(totalValue);
  const s = parseFloat(shares);
  if (!tv || !s || s === 0) return 0;
  return tv / s;
};

const getCategoryIcon = (category: string) => {
  const normalized = category.toLowerCase().replace(/[& ]+/g, '_').replace(/__+/g, '_');
  for (const [key, val] of Object.entries(CATEGORY_MAP)) {
    if (key === normalized || val.label.toLowerCase() === category.toLowerCase()) {
      return val.icon;
    }
  }
  return Briefcase;
};

const getCategoryKey = (category: string): string => {
  const normalized = category.toLowerCase().replace(/[& ]+/g, '_').replace(/__+/g, '_');
  for (const key of Object.keys(CATEGORY_MAP)) {
    if (key === normalized || CATEGORY_MAP[key].label.toLowerCase() === category.toLowerCase()) {
      return key;
    }
  }
  return 'all';
};

// ─── Sub-components ──────────────────────────────────────────────────────────

function StatCard({ label, value, icon: Icon }: { label: string; value: string; icon: typeof DollarSign }) {
  return (
    <div className="flex items-center gap-3 px-5 py-3">
      <div className="w-9 h-9 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
        <Icon size={16} className="text-[#D4AF37]/70" />
      </div>
      <div>
        <p className="text-[11px] text-gray-500 font-light tracking-wider uppercase">{label}</p>
        <p className="text-sm text-white font-light">{value}</p>
      </div>
    </div>
  );
}

function CategoryChip({
  categoryKey,
  isActive,
  onClick,
}: {
  categoryKey: string;
  isActive: boolean;
  onClick: () => void;
}) {
  const cat = CATEGORY_MAP[categoryKey];
  if (!cat) return null;
  const Icon = cat.icon;
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-2 rounded-full text-xs font-light tracking-wide
        border transition-all duration-500 ease-out whitespace-nowrap
        ${isActive
          ? 'border-[#D4AF37]/60 text-[#D4AF37] bg-[#D4AF37]/[0.06]'
          : 'border-white/10 text-gray-500 hover:text-gray-300 hover:border-white/20 bg-transparent'
        }
      `}
    >
      <Icon size={13} />
      {cat.label}
    </button>
  );
}

function AssetCard({
  asset,
  viewMode,
  onClick,
}: {
  asset: RwaListing;
  viewMode: 'grid' | 'list';
  onClick: () => void;
}) {
  const Icon = getCategoryIcon(asset.category);
  const pps = pricePerShare(asset.total_value_usd, asset.shares_available);
  const isCampaign = !!asset.campaign_id;

  if (viewMode === 'list') {
    return (
      <motion.div
        layout
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
        onClick={onClick}
        className="
          group flex items-center gap-6 px-6 py-5 cursor-pointer
          bg-gradient-to-r from-[#141420]/60 to-[#0d0d15]/60 backdrop-blur-xl
          border border-white/[0.06] rounded-xl
          hover:border-white/[0.12] hover:shadow-[0_8px_32px_rgba(212,175,55,0.06)]
          transition-all duration-500 ease-out
        "
      >
        <div className="w-11 h-11 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center flex-shrink-0">
          <Icon size={18} className="text-gray-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <h3 className="text-sm text-white font-light truncate">{asset.name}</h3>
            {asset.verified && <Shield size={12} className="text-emerald-500/80 flex-shrink-0" />}
          </div>
          <p className="text-xs text-gray-500 font-light truncate">{asset.symbol} &middot; {asset.category}</p>
        </div>
        {isCampaign ? (
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="w-32">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-gray-500 font-light">{(asset.progress_percent ?? 0).toFixed(0)}%</span>
                <span className="text-[10px] text-[#D4AF37] font-light">{formatCurrency(asset.raised_usd ?? 0)}</span>
              </div>
              <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#D4AF37] to-[#B8960C] rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(asset.progress_percent ?? 0, 100)}%` }}
                />
              </div>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded-full border border-[#D4AF37]/20 text-[#D4AF37]/80 bg-[#D4AF37]/[0.06]">
              {asset.contributor_count ?? 0} backers
            </span>
          </div>
        ) : (
          <>
            <div className="text-right flex-shrink-0">
              <p className="text-sm text-[#D4AF37] font-light">{formatCurrency(pps)}<span className="text-[10px] text-gray-500">/share</span></p>
              <p className="text-[11px] text-gray-500 font-light">{formatCurrency(asset.total_value_usd)} total</p>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              {asset.dividend_enabled && (
                <span className="text-[10px] px-2 py-0.5 rounded-full border border-emerald-500/20 text-emerald-400/80 bg-emerald-500/[0.06]">
                  Yield
                </span>
              )}
              {asset.kyc_required && (
                <span className="text-[10px] px-2 py-0.5 rounded-full border border-amber-500/20 text-amber-400/80 bg-amber-500/[0.06]">
                  KYC
                </span>
              )}
            </div>
          </>
        )}
        <ChevronRight size={16} className="text-gray-600 group-hover:text-gray-400 transition-colors duration-300 flex-shrink-0" />
      </motion.div>
    );
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
      onClick={onClick}
      className="
        group cursor-pointer flex flex-col
        bg-gradient-to-b from-[#141420]/80 to-[#0d0d15]/80 backdrop-blur-xl
        border border-white/[0.06] rounded-2xl overflow-hidden
        hover:border-white/[0.12] hover:shadow-[0_12px_40px_rgba(212,175,55,0.08)]
        hover:-translate-y-[2px]
        transition-all duration-500 ease-out
      "
    >
      {/* Card header */}
      <div className="px-5 pt-5 pb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] px-2.5 py-1 rounded-full border border-white/[0.08] text-gray-400 font-light tracking-wide">
            {asset.category}
          </span>
          {asset.verified && (
            <div className="flex items-center gap-1">
              <Shield size={11} className="text-emerald-500/80" />
              <span className="text-[10px] text-emerald-400/70 font-light">Verified</span>
            </div>
          )}
        </div>
        <div className="w-8 h-8 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
          <Icon size={15} className="text-gray-500 group-hover:text-[#D4AF37]/60 transition-colors duration-500" />
        </div>
      </div>

      {/* Card body */}
      <div className="px-5 pb-4 flex-1">
        <h3 className="text-[15px] text-white font-light leading-snug mb-1 tracking-wide">{asset.name}</h3>
        <p className="text-[11px] text-gray-500 font-light mb-0.5">{asset.symbol}</p>
        <p className="text-xs text-gray-500/80 font-light line-clamp-2 leading-relaxed mt-2">{asset.description}</p>
      </div>

      {/* Feature tags */}
      <div className="px-5 pb-3 flex flex-wrap gap-1.5">
        {asset.dividend_enabled && (
          <span className="text-[9px] px-2 py-0.5 rounded-full border border-emerald-500/15 text-emerald-400/70 bg-emerald-500/[0.04]">
            Dividends
          </span>
        )}
        {asset.kyc_required && (
          <span className="text-[9px] px-2 py-0.5 rounded-full border border-amber-500/15 text-amber-400/70 bg-amber-500/[0.04]">
            KYC Required
          </span>
        )}
        {Object.entries(asset.features || {}).filter(([, v]) => v).slice(0, 2).map(([key]) => (
          <span key={key} className="text-[9px] px-2 py-0.5 rounded-full border border-white/[0.06] text-gray-500 bg-white/[0.02]">
            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </span>
        ))}
      </div>

      {/* Divider */}
      <div className="mx-5 border-t border-white/[0.06]" />

      {/* Card footer */}
      {isCampaign ? (
        <>
          <div className="px-5 py-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-gray-500 font-light tracking-wider uppercase">
                {formatCurrency(asset.raised_usd ?? 0)} raised
              </span>
              <span className="text-[10px] text-gray-500 font-light">
                of {formatCurrency(asset.total_value_usd)}
              </span>
            </div>
            <div className="w-full h-2 bg-white/[0.06] rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-gradient-to-r from-[#D4AF37] to-[#B8960C] rounded-full transition-all duration-500"
                style={{ width: `${Math.min(asset.progress_percent ?? 0, 100)}%` }}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#D4AF37] font-light">{(asset.progress_percent ?? 0).toFixed(0)}% funded</span>
              <span className="text-[10px] text-gray-500 font-light flex items-center gap-1">
                <Users size={10} />
                {asset.contributor_count ?? 0} contributors
              </span>
            </div>
          </div>
          <div className="px-5 pb-4">
            <div className="w-full py-2.5 rounded-xl text-xs font-semibold tracking-wide text-center
              bg-gradient-to-r from-[#D4AF37]/20 to-[#B8960C]/20 text-[#D4AF37] border border-[#D4AF37]/20
              group-hover:from-[#D4AF37] group-hover:to-[#B8960C] group-hover:text-black
              transition-all duration-500">
              Contribute
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="px-5 py-4 flex items-end justify-between">
            <div>
              <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Price / Share</p>
              <p className="text-lg text-[#D4AF37] font-light">{formatCurrency(pps)}</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Total Value</p>
              <p className="text-sm text-white/80 font-light">{formatCurrency(asset.total_value_usd)}</p>
            </div>
          </div>
          <div className="px-5 pb-4 flex items-center justify-between">
            <p className="text-[10px] text-gray-600 font-light">
              {formatNumber(asset.shares_available)} shares available
            </p>
            <p className="text-[10px] text-gray-600 font-light flex items-center gap-1">
              <Clock size={10} />
              {timeAgo(asset.deployed_at)}
            </p>
          </div>
        </>
      )}
    </motion.div>
  );
}

function PortfolioCard({ holding }: { holding: PortfolioHolding }) {
  const gain = holding.currentValue - holding.costBasis;
  const gainPct = holding.costBasis > 0 ? (gain / holding.costBasis) * 100 : 0;
  const isPositive = gain >= 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="
        bg-gradient-to-b from-[#141420]/80 to-[#0d0d15]/80 backdrop-blur-xl
        border border-white/[0.06] rounded-2xl p-5
        hover:border-white/[0.12] transition-all duration-500 ease-out
      "
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm text-white font-light tracking-wide">{holding.name}</h3>
          <p className="text-[11px] text-gray-500 font-light">{holding.symbol}</p>
        </div>
        <div className={`flex items-center gap-1 text-xs font-light ${isPositive ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
          <ArrowUpRight size={12} className={!isPositive ? 'rotate-180' : ''} />
          {isPositive ? '+' : ''}{gainPct.toFixed(2)}%
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Shares Owned</p>
          <p className="text-sm text-white font-light">{formatNumber(holding.sharesOwned)}</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Current Value</p>
          <p className="text-sm text-[#D4AF37] font-light">{formatCurrency(holding.currentValue)}</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Cost Basis</p>
          <p className="text-sm text-gray-400 font-light">{formatCurrency(holding.costBasis)}</p>
        </div>
        <div>
          <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Yield Earned</p>
          <p className="text-sm text-emerald-400/80 font-light">{formatCurrency(holding.yieldEarned)}</p>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function RwaMarketplaceScreen() {
  const [listings, setListings] = useState<RwaListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedAsset, setSelectedAsset] = useState<RwaListing | null>(null);
  const [investAmount, setInvestAmount] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'marketplace' | 'portfolio'>('marketplace');
  const [portfolio, setPortfolio] = useState<PortfolioHolding[]>([]);
  const [sortBy, setSortBy] = useState<string>('newest');
  const [investLoading, setInvestLoading] = useState(false);
  const [investSuccess, setInvestSuccess] = useState(false);
  const [investError, setInvestError] = useState('');
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [showSortDropdown, setShowSortDropdown] = useState(false);
  const [crowdfundCampaign, setCrowdfundCampaign] = useState<any>(null);

  // ─── Data Fetching ───────────────────────────────────────────────────────

  const fetchListings = useCallback(async () => {
    setLoading(true);
    try {
      const categoryParam = selectedCategory !== 'all'
        ? `?category=${CATEGORY_MAP[selectedCategory]?.apiKey || selectedCategory}`
        : '';
      const response = await fetch(`/api/v1/contracts/rwa/marketplace${categoryParam}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: ApiResponse<RwaListing[]> = await response.json();
      if (data.success && data.data) {
        setListings(data.data);
      } else {
        setListings([]);
      }
    } catch (err) {
      console.error('[RWA] Failed to fetch marketplace listings:', err);
      setListings([]);
    } finally {
      setLoading(false);
    }
  }, [selectedCategory]);

  const fetchPortfolio = useCallback(async () => {
    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      setPortfolio([]);
      return;
    }
    setPortfolioLoading(true);
    try {
      const response = await fetch(`/api/v1/contracts/user/${walletAddress}/contracts`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: ApiResponse<any[]> = await response.json();
      if (data.success && data.data) {
        const rwaTypes = [
          'RwaToken', 'RealEstateToken', 'EquityToken', 'FixedIncomeToken',
          'CommodityToken', 'CarbonCreditToken', 'ArtCollectibleToken',
          'IPRevenueToken', 'PhysicalGoodsToken',
        ];
        const rwaHoldings: PortfolioHolding[] = data.data
          .filter((c: any) => rwaTypes.some(t => (c.type || c.contract_type || '').includes(t)))
          .map((c: any) => ({
            address: c.address || '',
            name: c.name || 'Unknown Asset',
            symbol: c.symbol || '',
            type: c.type || c.contract_type || '',
            sharesOwned: parseFloat(c.abaBalance || c.balance || '0'),
            currentValue: parseFloat(c.total_value_usd || '0'),
            costBasis: 0,
            yieldEarned: 0,
            deployedAt: c.deployedAt || c.deployed_at || 0,
          }));
        setPortfolio(rwaHoldings);
      } else {
        setPortfolio([]);
      }
    } catch (err) {
      console.error('[RWA] Failed to fetch portfolio:', err);
      setPortfolio([]);
    } finally {
      setPortfolioLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchListings();
  }, [fetchListings]);

  useEffect(() => {
    if (activeTab === 'portfolio') {
      fetchPortfolio();
    }
  }, [activeTab, fetchPortfolio]);

  // ─── Invest Handler ──────────────────────────────────────────────────────

  const handleInvest = useCallback(async () => {
    if (!selectedAsset || !investAmount) return;
    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) {
      setInvestError('Please connect your wallet first.');
      return;
    }

    const amount = parseFloat(investAmount);
    if (isNaN(amount) || amount <= 0) {
      setInvestError('Please enter a valid amount.');
      return;
    }
    if (amount > parseFloat(selectedAsset.shares_available)) {
      setInvestError('Amount exceeds available shares.');
      return;
    }

    setInvestLoading(true);
    setInvestError('');
    setInvestSuccess(false);

    try {
      const response = await fetch(`/api/v1/contracts/${selectedAsset.address}/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'transfer',
          to_address: walletAddress,
          amount: investAmount,
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.success) {
        setInvestSuccess(true);
        setInvestAmount('');
        setTimeout(() => {
          setInvestSuccess(false);
        }, 4000);
      } else {
        setInvestError(data.error || 'Transaction failed.');
      }
    } catch (err) {
      console.error('[RWA] Investment failed:', err);
      setInvestError('Network error. Please try again.');
    } finally {
      setInvestLoading(false);
    }
  }, [selectedAsset, investAmount]);

  // ─── Campaign Click Handler ──────────────────────────────────────────────

  const handleAssetClick = useCallback(async (asset: RwaListing) => {
    if (asset.campaign_id) {
      // Fetch full campaign object for the crowdfund modal
      try {
        const resp = await fetch(`/api/v1/contracts/listing/campaigns/${asset.campaign_id}`);
        if (resp.ok) {
          const data = await resp.json();
          if (data.success && data.data?.campaign) {
            setCrowdfundCampaign(data.data.campaign);
            return;
          }
        }
      } catch (err) {
        console.error('[RWA] Failed to fetch campaign details:', err);
      }
      // Fallback: construct a minimal campaign object from listing data
      setCrowdfundCampaign({
        campaign_id: asset.campaign_id,
        exchange_name: asset.name.replace(' Exchange Listing', ''),
        exchange_logo: '',
        target_usd: asset.target_usd_num || parseFloat(asset.total_value_usd) || 0,
        raised_usd: asset.raised_usd || 0,
        contributor_count: asset.contributor_count || 0,
        early_bird_slots: 0,
        early_bird_claimed: 0,
        status: asset.campaign_status || 'funding',
        tier: 'silver',
        description: asset.description,
        perks: {},
      });
    } else {
      setSelectedAsset(asset);
    }
  }, []);

  // ─── Filtering & Sorting ─────────────────────────────────────────────────

  const filteredListings = useMemo(() => {
    let result = [...listings];

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (a) =>
          a.name.toLowerCase().includes(q) ||
          a.symbol.toLowerCase().includes(q) ||
          a.category.toLowerCase().includes(q) ||
          a.description.toLowerCase().includes(q)
      );
    }

    switch (sortBy) {
      case 'value_high':
        result.sort((a, b) => parseFloat(b.total_value_usd) - parseFloat(a.total_value_usd));
        break;
      case 'value_low':
        result.sort((a, b) => parseFloat(a.total_value_usd) - parseFloat(b.total_value_usd));
        break;
      case 'yield':
        result.sort((a, b) => (b.dividend_enabled ? 1 : 0) - (a.dividend_enabled ? 1 : 0));
        break;
      case 'newest':
      default:
        result.sort((a, b) => b.deployed_at - a.deployed_at);
        break;
    }

    return result;
  }, [listings, searchQuery, sortBy]);

  // ─── Computed Stats ──────────────────────────────────────────────────────

  const stats = useMemo(() => {
    const totalMarketCap = listings.reduce((sum, a) => sum + parseFloat(a.total_value_usd || '0'), 0);
    const totalAssets = listings.length;
    const dividendCount = listings.filter((a) => a.dividend_enabled).length;
    const avgYieldPct = totalAssets > 0 ? ((dividendCount / totalAssets) * 100).toFixed(0) : '0';
    return {
      totalMarketCap: formatCurrency(totalMarketCap),
      totalAssets: totalAssets.toString(),
      dividendPct: `${avgYieldPct}% yield-bearing`,
      verifiedCount: listings.filter((a) => a.verified).length.toString(),
    };
  }, [listings]);

  // ─── Estimated Cost ──────────────────────────────────────────────────────

  const estimatedCost = useMemo(() => {
    if (!selectedAsset || !investAmount) return 0;
    const amount = parseFloat(investAmount);
    if (isNaN(amount)) return 0;
    return amount * pricePerShare(selectedAsset.total_value_usd, selectedAsset.shares_available);
  }, [selectedAsset, investAmount]);

  // ─── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* ─── Hero Section ─────────────────────────────────────────────── */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-[#0f0f18] via-[#0a0a0f] to-[#0a0a0f]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(212,175,55,0.03)_0%,transparent_70%)]" />
        <div className="relative max-w-7xl mx-auto px-6 pt-10 pb-6">
          <div className="flex items-end justify-between mb-8">
            <div>
              <p className="text-[11px] text-[#D4AF37]/60 font-light tracking-[0.3em] uppercase mb-3">
                Tokenized Investments
              </p>
              <h1 className="text-3xl text-white font-extralight tracking-[0.15em] uppercase">
                Real World Assets
              </h1>
              <p className="text-sm text-gray-500 font-light mt-2 max-w-md leading-relaxed">
                Invest in tokenized real-world assets. Fractional ownership of property, equity,
                bonds, commodities, and more — all on-chain.
              </p>
            </div>
            {/* Tab switcher */}
            <div className="flex items-center gap-1 bg-white/[0.03] border border-white/[0.06] rounded-xl p-1">
              <button
                onClick={() => setActiveTab('marketplace')}
                className={`
                  px-4 py-2 rounded-lg text-xs font-light tracking-wide transition-all duration-300
                  ${activeTab === 'marketplace'
                    ? 'bg-white/[0.06] text-white'
                    : 'text-gray-500 hover:text-gray-300'
                  }
                `}
              >
                <ShoppingCart size={13} className="inline-block mr-1.5 -mt-0.5" />
                Marketplace
              </button>
              <button
                onClick={() => setActiveTab('portfolio')}
                className={`
                  px-4 py-2 rounded-lg text-xs font-light tracking-wide transition-all duration-300
                  ${activeTab === 'portfolio'
                    ? 'bg-white/[0.06] text-white'
                    : 'text-gray-500 hover:text-gray-300'
                  }
                `}
              >
                <Briefcase size={13} className="inline-block mr-1.5 -mt-0.5" />
                My Portfolio
              </button>
            </div>
          </div>

          {/* Stats bar */}
          {activeTab === 'marketplace' && (
            <div className="flex items-center gap-0 bg-gradient-to-r from-[#141420]/50 to-[#0d0d15]/50 border border-white/[0.06] rounded-xl overflow-hidden">
              <StatCard label="Total Market Cap" value={stats.totalMarketCap} icon={DollarSign} />
              <div className="w-px h-10 bg-white/[0.06]" />
              <StatCard label="Listed Assets" value={stats.totalAssets} icon={BarChart3} />
              <div className="w-px h-10 bg-white/[0.06]" />
              <StatCard label="Yield Bearing" value={stats.dividendPct} icon={TrendingUp} />
              <div className="w-px h-10 bg-white/[0.06]" />
              <StatCard label="Verified" value={stats.verifiedCount} icon={Shield} />
            </div>
          )}
        </div>
      </div>

      {/* ─── Content Area ─────────────────────────────────────────────── */}
      <div className="max-w-7xl mx-auto px-6 pb-20">
        <AnimatePresence mode="wait">
          {activeTab === 'marketplace' ? (
            <motion.div
              key="marketplace"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {/* ─── Category Filter Bar ─────────────────────────────── */}
              <div className="mt-6 mb-5 flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                {Object.keys(CATEGORY_MAP).map((key) => (
                  <CategoryChip
                    key={key}
                    categoryKey={key}
                    isActive={selectedCategory === key}
                    onClick={() => setSelectedCategory(key)}
                  />
                ))}
              </div>

              {/* ─── Search, Sort, View Controls ─────────────────────── */}
              <div className="flex items-center gap-3 mb-6">
                {/* Search */}
                <div className="flex-1 relative">
                  <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-600" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search assets by name, symbol, or category..."
                    className="
                      w-full pl-10 pr-4 py-2.5 rounded-xl text-sm font-light
                      bg-white/[0.03] border border-white/[0.06] text-white
                      placeholder:text-gray-600
                      focus:outline-none focus:border-white/[0.12]
                      transition-all duration-300
                    "
                  />
                </div>

                {/* Sort dropdown */}
                <div className="relative">
                  <button
                    onClick={() => setShowSortDropdown(!showSortDropdown)}
                    className="
                      flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-light
                      bg-white/[0.03] border border-white/[0.06] text-gray-400
                      hover:border-white/[0.12] hover:text-gray-300
                      transition-all duration-300
                    "
                  >
                    <SortDesc size={14} />
                    {SORT_OPTIONS.find((o) => o.value === sortBy)?.label || 'Sort'}
                  </button>
                  <AnimatePresence>
                    {showSortDropdown && (
                      <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                        transition={{ duration: 0.15 }}
                        className="
                          absolute right-0 top-full mt-1 z-50 min-w-[180px]
                          bg-[#1a1a28] border border-white/[0.08] rounded-xl
                          shadow-[0_12px_48px_rgba(0,0,0,0.6)] overflow-hidden
                        "
                      >
                        {SORT_OPTIONS.map((opt) => (
                          <button
                            key={opt.value}
                            onClick={() => {
                              setSortBy(opt.value);
                              setShowSortDropdown(false);
                            }}
                            className={`
                              w-full px-4 py-2.5 text-left text-xs font-light transition-colors duration-200
                              ${sortBy === opt.value
                                ? 'text-[#D4AF37] bg-[#D4AF37]/[0.06]'
                                : 'text-gray-400 hover:text-white hover:bg-white/[0.04]'
                              }
                            `}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* View toggle */}
                <div className="flex items-center gap-0 bg-white/[0.03] border border-white/[0.06] rounded-xl overflow-hidden">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`
                      p-2.5 transition-all duration-300
                      ${viewMode === 'grid' ? 'bg-white/[0.06] text-white' : 'text-gray-600 hover:text-gray-400'}
                    `}
                  >
                    <Grid3X3 size={15} />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={`
                      p-2.5 transition-all duration-300
                      ${viewMode === 'list' ? 'bg-white/[0.06] text-white' : 'text-gray-600 hover:text-gray-400'}
                    `}
                  >
                    <List size={15} />
                  </button>
                </div>
              </div>

              {/* ─── Asset Grid / List ──────────────────────────────────── */}
              {loading ? (
                <div className="flex flex-col items-center justify-center py-32">
                  <Loader2 size={28} className="text-[#D4AF37]/40 animate-spin mb-4" />
                  <p className="text-sm text-gray-500 font-light tracking-wide">Loading marketplace...</p>
                </div>
              ) : filteredListings.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-32">
                  <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-5">
                    <Building size={24} className="text-gray-600" />
                  </div>
                  <h3 className="text-lg text-white font-extralight tracking-[0.1em] uppercase mb-2">
                    No Assets Listed
                  </h3>
                  <p className="text-sm text-gray-500 font-light max-w-sm text-center leading-relaxed mb-6">
                    {searchQuery
                      ? `No assets match "${searchQuery}". Try a different search term.`
                      : 'The marketplace is awaiting its first listing. Deploy a tokenized real-world asset to get started.'}
                  </p>
                  {!searchQuery && (
                    <a
                      href="#vittua-vm"
                      className="
                        inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-semibold
                        bg-gradient-to-r from-[#D4AF37] to-[#B8960C] text-black
                        hover:shadow-[0_4px_24px_rgba(212,175,55,0.25)]
                        transition-all duration-300
                      "
                    >
                      Deploy on VittuaVM
                      <ArrowUpRight size={13} />
                    </a>
                  )}
                </div>
              ) : (
                <AnimatePresence mode="popLayout">
                  {viewMode === 'grid' ? (
                    <motion.div
                      layout
                      className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4"
                    >
                      {filteredListings.map((asset) => (
                        <AssetCard
                          key={asset.address}
                          asset={asset}
                          viewMode="grid"
                          onClick={() => handleAssetClick(asset)}
                        />
                      ))}
                    </motion.div>
                  ) : (
                    <motion.div layout className="flex flex-col gap-2">
                      {filteredListings.map((asset) => (
                        <AssetCard
                          key={asset.address}
                          asset={asset}
                          viewMode="list"
                          onClick={() => handleAssetClick(asset)}
                        />
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              )}
            </motion.div>
          ) : (
            /* ─── Portfolio Tab ──────────────────────────────────────── */
            <motion.div
              key="portfolio"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="mt-8"
            >
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg text-white font-extralight tracking-[0.12em] uppercase">
                    Your RWA Holdings
                  </h2>
                  <p className="text-xs text-gray-500 font-light mt-1">
                    Tokenized real-world assets in your portfolio
                  </p>
                </div>
                <button
                  onClick={fetchPortfolio}
                  disabled={portfolioLoading}
                  className="
                    flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-light
                    bg-white/[0.03] border border-white/[0.06] text-gray-400
                    hover:border-white/[0.12] hover:text-gray-300
                    transition-all duration-300 disabled:opacity-50
                  "
                >
                  {portfolioLoading ? <Loader2 size={13} className="animate-spin" /> : <Star size={13} />}
                  Refresh
                </button>
              </div>

              {portfolioLoading ? (
                <div className="flex flex-col items-center justify-center py-24">
                  <Loader2 size={28} className="text-[#D4AF37]/40 animate-spin mb-4" />
                  <p className="text-sm text-gray-500 font-light">Loading portfolio...</p>
                </div>
              ) : portfolio.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-24">
                  <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-5">
                    <Briefcase size={24} className="text-gray-600" />
                  </div>
                  <h3 className="text-lg text-white font-extralight tracking-[0.1em] uppercase mb-2">
                    No Holdings Yet
                  </h3>
                  <p className="text-sm text-gray-500 font-light max-w-sm text-center leading-relaxed mb-6">
                    You haven't invested in any tokenized real-world assets yet. Browse the marketplace to get started.
                  </p>
                  <button
                    onClick={() => setActiveTab('marketplace')}
                    className="
                      inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-semibold
                      bg-gradient-to-r from-[#D4AF37] to-[#B8960C] text-black
                      hover:shadow-[0_4px_24px_rgba(212,175,55,0.25)]
                      transition-all duration-300
                    "
                  >
                    Browse Marketplace
                    <ChevronRight size={13} />
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                  {portfolio.map((holding) => (
                    <PortfolioCard key={holding.address} holding={holding} />
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ─── Asset Detail Modal ─────────────────────────────────────────── */}
      <AnimatePresence>
        {selectedAsset && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
            onClick={() => {
              setSelectedAsset(null);
              setInvestAmount('');
              setInvestError('');
              setInvestSuccess(false);
            }}
          >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, scale: 0.97, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 12 }}
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              onClick={(e) => e.stopPropagation()}
              className="
                relative w-full max-w-lg max-h-[90vh] overflow-y-auto
                bg-gradient-to-b from-[#161622] to-[#0e0e18]
                border border-white/[0.08] rounded-2xl
                shadow-[0_24px_80px_rgba(0,0,0,0.7)]
                scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent
              "
            >
              {/* Close button */}
              <button
                onClick={() => {
                  setSelectedAsset(null);
                  setInvestAmount('');
                  setInvestError('');
                  setInvestSuccess(false);
                }}
                className="
                  absolute top-4 right-4 z-10 w-8 h-8 rounded-lg
                  bg-white/[0.05] border border-white/[0.08]
                  flex items-center justify-center
                  text-gray-500 hover:text-white
                  transition-colors duration-200
                "
              >
                <X size={15} />
              </button>

              {/* Hero area */}
              <div className="relative px-6 pt-6 pb-5">
                <div className="absolute inset-0 bg-gradient-to-b from-[#D4AF37]/[0.02] to-transparent rounded-t-2xl" />
                <div className="relative">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center flex-shrink-0">
                      {(() => {
                        const Icon = getCategoryIcon(selectedAsset.category);
                        return <Icon size={24} className="text-[#D4AF37]/60" />;
                      })()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h2 className="text-xl text-white font-extralight tracking-[0.08em]">
                          {selectedAsset.name}
                        </h2>
                        {selectedAsset.verified && (
                          <Shield size={14} className="text-emerald-500/80 flex-shrink-0" />
                        )}
                      </div>
                      <p className="text-sm text-gray-500 font-light">
                        {selectedAsset.symbol} &middot; {selectedAsset.category}
                      </p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400/80 font-light leading-relaxed">
                    {selectedAsset.description}
                  </p>
                </div>
              </div>

              <div className="mx-6 border-t border-white/[0.06]" />

              {/* Key Metrics */}
              <div className="px-6 py-5">
                <p className="text-[10px] text-gray-600 font-light tracking-[0.2em] uppercase mb-3">
                  Key Metrics
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl px-4 py-3">
                    <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Total Value</p>
                    <p className="text-base text-white font-light">{formatCurrency(selectedAsset.total_value_usd)}</p>
                  </div>
                  <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl px-4 py-3">
                    <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Shares Available</p>
                    <p className="text-base text-white font-light">{formatNumber(selectedAsset.shares_available)}</p>
                  </div>
                  <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl px-4 py-3">
                    <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Price / Share</p>
                    <p className="text-base text-[#D4AF37] font-light">
                      {formatCurrency(pricePerShare(selectedAsset.total_value_usd, selectedAsset.shares_available))}
                    </p>
                  </div>
                  <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl px-4 py-3">
                    <p className="text-[10px] text-gray-600 font-light tracking-wider uppercase mb-0.5">Listed</p>
                    <p className="text-base text-white font-light">{formatTimestamp(selectedAsset.deployed_at)}</p>
                  </div>
                </div>
              </div>

              {/* Features & Compliance */}
              <div className="px-6 pb-4">
                <p className="text-[10px] text-gray-600 font-light tracking-[0.2em] uppercase mb-3">
                  Features & Compliance
                </p>
                <div className="flex flex-wrap gap-2">
                  {selectedAsset.kyc_required && (
                    <span className="text-[10px] px-3 py-1.5 rounded-lg border border-amber-500/20 text-amber-400/80 bg-amber-500/[0.06] font-light">
                      KYC Required
                    </span>
                  )}
                  {selectedAsset.dividend_enabled && (
                    <span className="text-[10px] px-3 py-1.5 rounded-lg border border-emerald-500/20 text-emerald-400/80 bg-emerald-500/[0.06] font-light">
                      Dividends Enabled
                    </span>
                  )}
                  {selectedAsset.verified && (
                    <span className="text-[10px] px-3 py-1.5 rounded-lg border border-emerald-500/20 text-emerald-400/80 bg-emerald-500/[0.06] font-light flex items-center gap-1">
                      <Shield size={10} /> Verified Asset
                    </span>
                  )}
                  {Object.entries(selectedAsset.features || {})
                    .filter(([, v]) => v)
                    .map(([key]) => (
                      <span
                        key={key}
                        className="text-[10px] px-3 py-1.5 rounded-lg border border-white/[0.06] text-gray-400 bg-white/[0.02] font-light"
                      >
                        {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                      </span>
                    ))}
                </div>
              </div>

              <div className="mx-6 border-t border-white/[0.06]" />

              {/* Investment Form */}
              <div className="px-6 py-5">
                <p className="text-[10px] text-gray-600 font-light tracking-[0.2em] uppercase mb-3">
                  Invest
                </p>

                {/* Amount input */}
                <div className="relative mb-3">
                  <input
                    type="number"
                    min="0"
                    step="1"
                    value={investAmount}
                    onChange={(e) => {
                      setInvestAmount(e.target.value);
                      setInvestError('');
                    }}
                    placeholder="Number of shares"
                    className="
                      w-full px-4 py-3 pr-16 rounded-xl text-sm font-light
                      bg-white/[0.03] border border-white/[0.06] text-white
                      placeholder:text-gray-600
                      focus:outline-none focus:border-[#D4AF37]/30
                      transition-all duration-300
                      [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none
                    "
                  />
                  <button
                    onClick={() => setInvestAmount(selectedAsset.shares_available)}
                    className="
                      absolute right-2 top-1/2 -translate-y-1/2
                      px-2.5 py-1 rounded-lg text-[10px] font-semibold
                      text-[#D4AF37] bg-[#D4AF37]/[0.08] border border-[#D4AF37]/20
                      hover:bg-[#D4AF37]/[0.15]
                      transition-all duration-200
                    "
                  >
                    MAX
                  </button>
                </div>

                {/* Estimated cost */}
                {investAmount && parseFloat(investAmount) > 0 && (
                  <div className="flex items-center justify-between mb-4 px-1">
                    <span className="text-[11px] text-gray-500 font-light">Estimated Cost</span>
                    <span className="text-sm text-white font-light">{formatCurrency(estimatedCost)}</span>
                  </div>
                )}

                {/* Error message */}
                <AnimatePresence>
                  {investError && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mb-3 flex items-center gap-2 px-3 py-2.5 rounded-xl bg-red-500/[0.06] border border-red-500/20"
                    >
                      <AlertCircle size={13} className="text-red-400/80 flex-shrink-0" />
                      <p className="text-[11px] text-red-400/80 font-light">{investError}</p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Success message */}
                <AnimatePresence>
                  {investSuccess && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mb-3 flex items-center gap-2 px-3 py-2.5 rounded-xl bg-emerald-500/[0.06] border border-emerald-500/20"
                    >
                      <CheckCircle2 size={13} className="text-emerald-400/80 flex-shrink-0" />
                      <p className="text-[11px] text-emerald-400/80 font-light">
                        Investment submitted successfully. Shares will appear in your portfolio.
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Invest button */}
                <button
                  onClick={handleInvest}
                  disabled={investLoading || !investAmount || parseFloat(investAmount) <= 0}
                  className="
                    w-full py-3.5 rounded-xl text-sm font-semibold tracking-wide
                    bg-gradient-to-r from-[#D4AF37] to-[#B8960C] text-black
                    hover:shadow-[0_4px_24px_rgba(212,175,55,0.3)]
                    disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none
                    transition-all duration-300
                    flex items-center justify-center gap-2
                  "
                >
                  {investLoading ? (
                    <>
                      <Loader2 size={15} className="animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      Invest Now
                      <ArrowUpRight size={15} />
                    </>
                  )}
                </button>

                {/* Disclaimer */}
                <p className="text-[9px] text-gray-600 font-light text-center mt-3 leading-relaxed px-2">
                  Investment in tokenized assets carries risk. Verify asset backing and compliance
                  documentation before investing. Past performance is not indicative of future results.
                </p>
              </div>

              {/* Contract address */}
              <div className="mx-6 border-t border-white/[0.06]" />
              <div className="px-6 py-4 flex items-center justify-between">
                <div>
                  <p className="text-[9px] text-gray-600 font-light tracking-wider uppercase">Contract Address</p>
                  <p className="text-[11px] text-gray-500 font-mono mt-0.5">
                    {selectedAsset.address.length > 20
                      ? `${selectedAsset.address.slice(0, 10)}...${selectedAsset.address.slice(-10)}`
                      : selectedAsset.address}
                  </p>
                </div>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(selectedAsset.address);
                  }}
                  className="
                    px-3 py-1.5 rounded-lg text-[10px] font-light
                    bg-white/[0.03] border border-white/[0.06] text-gray-500
                    hover:text-gray-300 hover:border-white/[0.12]
                    transition-all duration-200
                  "
                >
                  Copy
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Crowdfund Modal (for campaign listings) ──────────────────────── */}
      {crowdfundCampaign && (
        <XListCrowdfundModal
          campaign={crowdfundCampaign}
          onClose={() => setCrowdfundCampaign(null)}
          walletAddress={localStorage.getItem('walletAddress') || ''}
        />
      )}

      {/* ─── Custom Scrollbar Styles ────────────────────────────────────── */}
      <style>{`
        .scrollbar-thin::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        .scrollbar-thin::-webkit-scrollbar-track {
          background: transparent;
        }
        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: rgba(192, 192, 192, 0.15);
          border-radius: 4px;
        }
        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: rgba(192, 192, 192, 0.3);
        }
        .scrollbar-thumb-white\/10::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
        }
        .scrollbar-track-transparent::-webkit-scrollbar-track {
          background: transparent;
        }
      `}</style>
    </div>
  );
}
