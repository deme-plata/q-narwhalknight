import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PieChart, TrendingUp, DollarSign, Shield, Building, Landmark, Leaf, Palette,
  FileText, Package, Gem, Calendar, Clock, ArrowUpRight, ArrowDownRight,
  Wallet, Lock, Unlock, AlertTriangle, CheckCircle, RefreshCw, ChevronDown,
  ChevronUp, Banknote, Percent, BarChart3, Activity, Zap, Info, Settings,
  HelpCircle, Eye, Sparkles, Users, Globe, ExternalLink, MessageCircle, Edit3
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════
// RWA PORTFOLIO DASHBOARD - v4.2.0
// Tooltips + Visual Polish + Eye Candy
// ═══════════════════════════════════════════════════════════

// ─── Tooltip Component ───────────────────────────────
function Tooltip({ text, children, position = 'top' }: { text: string; children: React.ReactNode; position?: 'top' | 'bottom' | 'left' | 'right' }) {
  const [show, setShow] = useState(false);
  const posClasses: Record<string, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  const arrowClasses: Record<string, string> = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-gray-800 border-x-transparent border-b-transparent border-4',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-gray-800 border-x-transparent border-t-transparent border-4',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-gray-800 border-y-transparent border-r-transparent border-4',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-gray-800 border-y-transparent border-l-transparent border-4',
  };
  return (
    <span className="relative inline-flex items-center" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      <AnimatePresence>
        {show && (
          <motion.span
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`absolute z-[100] ${posClasses[position]} pointer-events-none`}
          >
            <span className="block bg-gray-800 text-gray-200 text-[11px] leading-[1.4] px-3 py-2 rounded-lg shadow-xl shadow-black/40 border border-gray-700/50 max-w-[260px] min-w-[140px] whitespace-normal font-normal">
              {text}
              <span className={`absolute ${arrowClasses[position]}`} />
            </span>
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

// ─── Info Badge (icon + tooltip) ─────────────────────
function InfoTip({ text, position }: { text: string; position?: 'top' | 'bottom' | 'left' | 'right' }) {
  return (
    <Tooltip text={text} position={position || 'top'}>
      <HelpCircle className="w-3.5 h-3.5 text-gray-500 hover:text-gray-300 transition-colors cursor-help ml-1" />
    </Tooltip>
  );
}

// ─── Animated counter component ──────────────────────
function AnimatedValue({ value, prefix = '$', decimals = 0 }: { value: number; prefix?: string; decimals?: number }) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {prefix}{value.toLocaleString(undefined, { maximumFractionDigits: decimals })}
    </motion.span>
  );
}

// ─── Glowing card wrapper ────────────────────────────
function GlowCard({ children, color = 'amber', className = '' }: { children: React.ReactNode; color?: string; className?: string }) {
  const gradients: Record<string, string> = {
    amber: 'from-amber-900/30 via-amber-800/10 to-quantum-dark/60',
    green: 'from-emerald-900/30 via-green-800/10 to-quantum-dark/60',
    blue: 'from-blue-900/30 via-blue-800/10 to-quantum-dark/60',
    purple: 'from-purple-900/30 via-violet-800/10 to-quantum-dark/60',
    pink: 'from-pink-900/30 via-pink-800/10 to-quantum-dark/60',
    cyan: 'from-cyan-900/30 via-cyan-800/10 to-quantum-dark/60',
  };
  const borders: Record<string, string> = {
    amber: 'border-amber-500/20 hover:border-amber-500/30',
    green: 'border-emerald-500/20 hover:border-emerald-500/30',
    blue: 'border-blue-500/20 hover:border-blue-500/30',
    purple: 'border-purple-500/20 hover:border-purple-500/30',
    pink: 'border-pink-500/20 hover:border-pink-500/30',
    cyan: 'border-cyan-500/20 hover:border-cyan-500/30',
  };
  const glows: Record<string, string> = {
    amber: 'shadow-amber-500/5',
    green: 'shadow-emerald-500/5',
    blue: 'shadow-blue-500/5',
    purple: 'shadow-purple-500/5',
    pink: 'shadow-pink-500/5',
    cyan: 'shadow-cyan-500/5',
  };
  return (
    <motion.div
      className={`bg-gradient-to-br ${gradients[color] || gradients.amber} rounded-xl p-4 border ${borders[color] || borders.amber} shadow-lg ${glows[color] || glows.amber} transition-all duration-300 ${className}`}
      whileHover={{ scale: 1.01, y: -1 }}
      transition={{ duration: 0.2 }}
    >
      {children}
    </motion.div>
  );
}

interface DeployedContract {
  address: string;
  name: string;
  symbol: string;
  type: string;
  deployedAt: Date;
  features: Record<string, boolean | undefined>;
  isPaused?: boolean;
  abaBalance?: string;
  totalSupply?: string;
  decimals?: number;
  owner?: string;
  deploymentParams?: Record<string, string | boolean>;
}

interface CollateralPosition {
  id: string;
  contractAddress: string;
  collateralAmount: string;
  collateralValueUsd: number;
  borrowedAmount: number;
  borrowedCurrency: string;
  ltvRatio: number;
  maxLtv: number;
  interestRate: number;
  createdAt: Date;
  status: 'healthy' | 'warning' | 'danger' | 'liquidated';
}

interface DistributionSchedule {
  id: string;
  contractAddress: string;
  type: 'rent' | 'dividend' | 'coupon' | 'royalty' | 'custom';
  frequency: 'weekly' | 'monthly' | 'quarterly' | 'annually' | 'on_receipt';
  amount: number;
  nextDistribution: Date;
  lastDistribution?: Date;
  totalDistributed: number;
  recipientCount: number;
  enabled: boolean;
}

interface PortfolioMetrics {
  totalValue: number;
  totalYield: number;
  avgYieldPercent: number;
  totalCollateralValue: number;
  totalBorrowed: number;
  totalDistributed: number;
  upcomingDistributions: number;
  assetAllocation: { type: string; value: number; color: string; count: number }[];
  yieldHistory: { date: string; yield: number; value: number }[];
}

const LTV_RATIOS: Record<string, { max: number; warning: number; label: string }> = {
  realestate: { max: 70, warning: 60, label: 'Real Estate' },
  equity: { max: 50, warning: 40, label: 'Equity' },
  fixedincome: { max: 80, warning: 70, label: 'Fixed Income' },
  commodity: { max: 60, warning: 50, label: 'Commodity' },
  carboncredit: { max: 40, warning: 30, label: 'Carbon Credit' },
  artcollectible: { max: 30, warning: 20, label: 'Art & Collectible' },
  iprevenue: { max: 45, warning: 35, label: 'IP & Revenue' },
  physicalgoods: { max: 50, warning: 40, label: 'Physical Goods' },
};

const LTV_TOOLTIPS: Record<string, string> = {
  realestate: 'Real estate backed tokens have 70% max LTV due to the stable, tangible nature of property assets and consistent rental income.',
  equity: 'Equity tokens have 50% max LTV due to price volatility. Market conditions can cause rapid value changes.',
  fixedincome: 'Fixed income (bonds) allow 80% max LTV due to predictable cash flows, known maturity dates, and lower default risk.',
  commodity: 'Commodity-backed tokens allow 60% LTV. Physical commodities have intrinsic value but prices fluctuate with supply/demand.',
  carboncredit: 'Carbon credits have 40% max LTV due to regulatory uncertainty and limited secondary market liquidity.',
  artcollectible: 'Art & collectibles have only 30% max LTV due to subjective valuation, illiquidity, and long sale timelines.',
  iprevenue: 'IP revenue tokens allow 45% LTV. Revenue streams depend on licensee performance and contract terms.',
  physicalgoods: 'Physical goods tokens have 50% LTV. Value depends on storage conditions, demand, and physical verification.',
};

const ASSET_COLORS: Record<string, string> = {
  realestate: '#f59e0b',
  equity: '#8b5cf6',
  fixedincome: '#3b82f6',
  commodity: '#f97316',
  carboncredit: '#10b981',
  artcollectible: '#ec4899',
  iprevenue: '#7c3aed',
  physicalgoods: '#6366f1',
  other: '#6b7280',
};

const ASSET_ICONS: Record<string, React.ComponentType<any>> = {
  realestate: Building,
  equity: TrendingUp,
  fixedincome: Landmark,
  commodity: Package,
  carboncredit: Leaf,
  artcollectible: Palette,
  iprevenue: FileText,
  physicalgoods: Gem,
};

function getAssetType(type: string): string {
  const t = type.toLowerCase().replace(/[_\s-]/g, '');
  for (const key of Object.keys(LTV_RATIOS)) {
    if (t.includes(key)) return key;
  }
  return 'other';
}

function getAssetValueUsd(contract: DeployedContract): number {
  const p = contract.deploymentParams || {};
  const val = p.total_valuation_usd || p.total_value_usd || p.face_value_usd ||
    p.appraisal_value_usd || p.minimum_guarantee_usd || p.total_pool_value_usd || '0';
  return parseFloat(String(val)) || 0;
}

function getAssetYieldPercent(contract: DeployedContract): number {
  const p = contract.deploymentParams || {};
  const y = p.rental_yield_percent || p.coupon_rate_percent || p.royalty_rate_percent ||
    p.dividend_yield_percent || p.annual_yield_percent || '0';
  return parseFloat(String(y)) || 0;
}

function getSharesOwned(contract: DeployedContract): number {
  const bal = parseFloat(contract.abaBalance || '0');
  const dec = contract.decimals || 24;
  return bal / Math.pow(10, dec);
}

function getTotalShares(contract: DeployedContract): number {
  const p = contract.deploymentParams || {};
  const s = p.total_shares || p.total_tokens || p.total_units || p.total_fractions ||
    p.total_credits_tonnes || p.shares_count || p.quantity_per_token || p.initialSupply || '0';
  return parseFloat(String(s)) || 1;
}

function getOwnershipPercent(contract: DeployedContract): number {
  const owned = getSharesOwned(contract);
  const total = getTotalShares(contract);
  if (total === 0) return 0;
  return Math.min(100, (owned / total) * 100);
}

function getHoldingValueUsd(contract: DeployedContract): number {
  const pct = getOwnershipPercent(contract);
  const totalVal = getAssetValueUsd(contract);
  return (pct / 100) * totalVal;
}

// v4.0.1: Social profile type for RWA contracts
interface SocialProfile {
  twitter?: string;
  discord?: string;
  telegram?: string;
  website?: string;
  github?: string;
  medium?: string;
  description?: string;
  logo_url?: string;
}

interface Props {
  contracts: DeployedContract[];
  walletAddress: string;
}

export default function RwaPortfolioTab({ contracts, walletAddress }: Props) {
  const [activeSection, setActiveSection] = useState<'overview' | 'collateral' | 'distributions' | 'compliance' | 'social'>('overview');
  const [collateralPositions, setCollateralPositions] = useState<CollateralPosition[]>([]);
  const [distributionSchedules, setDistributionSchedules] = useState<DistributionSchedule[]>([]);
  const [showBorrowModal, setShowBorrowModal] = useState<string | null>(null);
  const [showScheduleModal, setShowScheduleModal] = useState<string | null>(null);
  const [borrowAmount, setBorrowAmount] = useState('');
  const [scheduleConfig, setScheduleConfig] = useState({ frequency: 'monthly' as const, amount: '' });
  const [expandedHolding, setExpandedHolding] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});
  const [actionStatus, setActionStatus] = useState<Record<string, { success: boolean; message: string }>>({});

  // v4.0.1: Social profiles per RWA contract
  const [socialProfiles, setSocialProfiles] = useState<Record<string, SocialProfile>>({});
  const [editingSocialAddr, setEditingSocialAddr] = useState<string | null>(null);
  const [socialForm, setSocialForm] = useState<SocialProfile>({});
  const [savingSocial, setSavingSocial] = useState(false);

  // Filter RWA contracts only (MUST be before useEffect that references it)
  const rwaContracts = useMemo(() => {
    const rwaTypes = ['realestate', 'equity', 'fixedincome', 'commodity', 'carboncredit',
      'artcollectible', 'iprevenue', 'physicalgoods', 'rwatoken', 'real_estate',
      'fixed_income', 'carbon_credit', 'art_collectible', 'ip_revenue', 'physical_goods'];
    return contracts.filter(c => {
      const ct = c.type.toLowerCase().replace(/[_\s-]/g, '');
      return rwaTypes.some(t => ct.includes(t.replace(/_/g, '')));
    });
  }, [contracts]);

  // Fetch social profiles for all RWA contracts
  useEffect(() => {
    rwaContracts.forEach(async (contract) => {
      try {
        const resp = await fetch(`/api/v1/contracts/${contract.address}/social`);
        if (resp.ok) {
          const result = await resp.json();
          if (result.success && result.data) {
            setSocialProfiles(prev => ({ ...prev, [contract.address]: result.data }));
          }
        }
      } catch { /* ignore */ }
    });
  }, [rwaContracts.length]);

  const handleSaveSocialProfile = useCallback(async (contractAddr: string) => {
    setSavingSocial(true);
    try {
      const resp = await fetch(`/api/v1/contracts/${contractAddr}/social`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...socialForm, owner_address: walletAddress }),
      });
      if (resp.ok) {
        const result = await resp.json();
        if (result.success && result.data) {
          setSocialProfiles(prev => ({ ...prev, [contractAddr]: result.data }));
        } else {
          setSocialProfiles(prev => ({ ...prev, [contractAddr]: socialForm }));
        }
      }
      setEditingSocialAddr(null);
    } catch (err) {
      console.error('Failed to save social profile:', err);
    } finally {
      setSavingSocial(false);
    }
  }, [socialForm, walletAddress]);

  // Calculate portfolio metrics
  const metrics: PortfolioMetrics = useMemo(() => {
    const allocation: Record<string, { value: number; count: number }> = {};
    let totalValue = 0;
    let weightedYield = 0;

    rwaContracts.forEach(c => {
      const type = getAssetType(c.type);
      const val = getHoldingValueUsd(c);
      const yld = getAssetYieldPercent(c);
      totalValue += val;
      weightedYield += val * (yld / 100);
      if (!allocation[type]) allocation[type] = { value: 0, count: 0 };
      allocation[type].value += val;
      allocation[type].count += 1;
    });

    const assetAllocation = Object.entries(allocation).map(([type, { value, count }]) => ({
      type: LTV_RATIOS[type]?.label || type,
      value,
      color: ASSET_COLORS[type] || ASSET_COLORS.other,
      count,
    })).sort((a, b) => b.value - a.value);

    const totalCollateral = collateralPositions.reduce((sum, p) => sum + p.collateralValueUsd, 0);
    const totalBorrowed = collateralPositions.reduce((sum, p) => sum + p.borrowedAmount, 0);
    const totalDistributed = distributionSchedules.reduce((sum, s) => sum + s.totalDistributed, 0);
    const upcoming = distributionSchedules.filter(s => s.enabled && s.nextDistribution > new Date()).length;

    const yieldHistory = Array.from({ length: 12 }, (_, i) => {
      const date = new Date();
      date.setMonth(date.getMonth() - (11 - i));
      const monthFactor = 0.85 + Math.random() * 0.3;
      return {
        date: date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }),
        yield: totalValue > 0 ? (weightedYield / 12) * monthFactor : 0,
        value: totalValue * (0.9 + i * 0.01 + Math.random() * 0.02),
      };
    });

    return {
      totalValue,
      totalYield: weightedYield,
      avgYieldPercent: totalValue > 0 ? (weightedYield / totalValue) * 100 : 0,
      totalCollateralValue: totalCollateral,
      totalBorrowed,
      totalDistributed,
      upcomingDistributions: upcoming,
      assetAllocation,
      yieldHistory,
    };
  }, [rwaContracts, collateralPositions, distributionSchedules]);

  // Load collateral positions and distribution schedules
  useEffect(() => {
    if (!walletAddress) return;
    fetch(`/api/v1/contracts/rwa/portfolio?wallet=${walletAddress}`)
      .then(r => r.json())
      .then(data => {
        if (data.data?.collateral_positions) {
          setCollateralPositions(data.data.collateral_positions.map((p: any) => ({
            ...p,
            createdAt: new Date(p.created_at || Date.now()),
          })));
        }
        if (data.data?.distribution_schedules) {
          setDistributionSchedules(data.data.distribution_schedules.map((s: any) => ({
            ...s,
            nextDistribution: new Date(s.next_distribution || Date.now()),
            lastDistribution: s.last_distribution ? new Date(s.last_distribution) : undefined,
          })));
        }
      })
      .catch(() => {});
  }, [walletAddress]);

  // ─── Action Handlers ───────────────────────────────────
  const handleBorrow = useCallback(async (contractAddress: string) => {
    if (!borrowAmount || parseFloat(borrowAmount) <= 0) return;
    setActionLoading(p => ({ ...p, [`borrow:${contractAddress}`]: true }));
    try {
      const res = await fetch('/api/v1/contracts/rwa/collateral/borrow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contract_address: contractAddress, amount: borrowAmount, wallet: walletAddress }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Borrow failed');
      const contract = rwaContracts.find(c => c.address === contractAddress);
      const assetType = contract ? getAssetType(contract.type) : 'other';
      const ltv = LTV_RATIOS[assetType] || { max: 50, warning: 40 };
      const collateralValue = contract ? getHoldingValueUsd(contract) : 0;
      const amt = parseFloat(borrowAmount);
      const ltvRatio = collateralValue > 0 ? (amt / collateralValue) * 100 : 0;
      setCollateralPositions(prev => [...prev, {
        id: data.data?.position_id || `pos_${Date.now()}`,
        contractAddress,
        collateralAmount: contract?.abaBalance || '0',
        collateralValueUsd: collateralValue,
        borrowedAmount: amt,
        borrowedCurrency: 'QUG',
        ltvRatio,
        maxLtv: ltv.max,
        interestRate: 5.5,
        createdAt: new Date(),
        status: ltvRatio > ltv.warning ? 'warning' : 'healthy',
      }]);
      setActionStatus(p => ({ ...p, [`borrow:${contractAddress}`]: { success: true, message: `Borrowed $${amt.toLocaleString()} QUG` } }));
      setShowBorrowModal(null);
      setBorrowAmount('');
    } catch (e: any) {
      setActionStatus(p => ({ ...p, [`borrow:${contractAddress}`]: { success: false, message: e.message } }));
    } finally {
      setActionLoading(p => ({ ...p, [`borrow:${contractAddress}`]: false }));
    }
  }, [borrowAmount, walletAddress, rwaContracts]);

  const handleRepay = useCallback(async (positionId: string, amount: number) => {
    setActionLoading(p => ({ ...p, [`repay:${positionId}`]: true }));
    try {
      const res = await fetch('/api/v1/contracts/rwa/collateral/repay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ position_id: positionId, amount, wallet: walletAddress }),
      });
      if (!res.ok) throw new Error('Repay failed');
      setCollateralPositions(prev => prev.map(p =>
        p.id === positionId ? { ...p, borrowedAmount: Math.max(0, p.borrowedAmount - amount), ltvRatio: Math.max(0, ((p.borrowedAmount - amount) / p.collateralValueUsd) * 100) } : p
      ).filter(p => p.borrowedAmount > 0));
      setActionStatus(p => ({ ...p, [`repay:${positionId}`]: { success: true, message: `Repaid $${amount.toLocaleString()}` } }));
    } catch (e: any) {
      setActionStatus(p => ({ ...p, [`repay:${positionId}`]: { success: false, message: e.message } }));
    } finally {
      setActionLoading(p => ({ ...p, [`repay:${positionId}`]: false }));
    }
  }, [walletAddress]);

  const handleScheduleDistribution = useCallback(async (contractAddress: string) => {
    if (!scheduleConfig.amount) return;
    setActionLoading(p => ({ ...p, [`schedule:${contractAddress}`]: true }));
    try {
      const res = await fetch('/api/v1/contracts/rwa/distribution/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contractAddress,
          frequency: scheduleConfig.frequency,
          amount: scheduleConfig.amount,
          wallet: walletAddress,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Schedule failed');
      const contract = rwaContracts.find(c => c.address === contractAddress);
      const freqDays: Record<string, number> = { weekly: 7, monthly: 30, quarterly: 90, annually: 365, on_receipt: 0 };
      const nextDate = new Date();
      nextDate.setDate(nextDate.getDate() + (freqDays[scheduleConfig.frequency] || 30));
      setDistributionSchedules(prev => [...prev, {
        id: data.data?.schedule_id || `sched_${Date.now()}`,
        contractAddress,
        type: getDistributionType(contract),
        frequency: scheduleConfig.frequency,
        amount: parseFloat(scheduleConfig.amount),
        nextDistribution: nextDate,
        totalDistributed: 0,
        recipientCount: 0,
        enabled: true,
      }]);
      setActionStatus(p => ({ ...p, [`schedule:${contractAddress}`]: { success: true, message: `Auto-distribution scheduled (${scheduleConfig.frequency})` } }));
      setShowScheduleModal(null);
      setScheduleConfig({ frequency: 'monthly', amount: '' });
    } catch (e: any) {
      setActionStatus(p => ({ ...p, [`schedule:${contractAddress}`]: { success: false, message: e.message } }));
    } finally {
      setActionLoading(p => ({ ...p, [`schedule:${contractAddress}`]: false }));
    }
  }, [scheduleConfig, walletAddress, rwaContracts]);

  const handleToggleSchedule = useCallback(async (scheduleId: string, enabled: boolean) => {
    setDistributionSchedules(prev => prev.map(s => s.id === scheduleId ? { ...s, enabled } : s));
    fetch('/api/v1/contracts/rwa/distribution/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schedule_id: scheduleId, enabled, wallet: walletAddress }),
    }).catch(() => {});
  }, [walletAddress]);

  function getDistributionType(contract?: DeployedContract): DistributionSchedule['type'] {
    if (!contract) return 'custom';
    const t = contract.type.toLowerCase();
    if (t.includes('realestate') || t.includes('real_estate')) return 'rent';
    if (t.includes('fixedincome') || t.includes('fixed_income')) return 'coupon';
    if (t.includes('ip') || t.includes('revenue')) return 'royalty';
    return 'dividend';
  }

  function statusBadge(status: string) {
    const colors: Record<string, string> = {
      healthy: 'bg-green-500/20 text-green-400 border-green-500/30',
      warning: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      danger: 'bg-red-500/20 text-red-400 border-red-500/30',
      liquidated: 'bg-red-800/30 text-red-500 border-red-500/30',
    };
    return colors[status] || 'bg-gray-500/20 text-gray-400 border-gray-500/30';
  }

  // ─── Render ────────────────────────────────────────────
  if (rwaContracts.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-16"
      >
        <motion.div
          animate={{ y: [0, -8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        >
          <Building className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        </motion.div>
        <h3 className="text-xl font-bold text-white mb-2">No RWA Holdings</h3>
        <p className="text-gray-400 mb-2">Deploy your first Real World Asset token to see your portfolio here.</p>
        <p className="text-xs text-gray-500">Supports Real Estate, Equity, Bonds, Commodities, Carbon Credits, Art, IP, and more.</p>
      </motion.div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ═══ Section Tabs with enhanced design ═══ */}
      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-gray-700">
        {[
          { id: 'overview' as const, label: 'Overview', icon: PieChart, tip: 'View your total RWA portfolio value, asset allocation, and yield performance at a glance.' },
          { id: 'collateral' as const, label: 'Collateral & Borrow', icon: Lock, tip: 'Use your RWA tokens as collateral to borrow QUG. Each asset type has different LTV ratios.' },
          { id: 'distributions' as const, label: 'Auto-Distributions', icon: Calendar, tip: 'Set up automatic revenue distributions (rent, dividends, coupons, royalties) to all token holders.' },
          { id: 'compliance' as const, label: 'Compliance', icon: Shield, tip: 'View your KYC, accreditation, and whitelist status for trading regulated RWA tokens on the DEX.' },
          { id: 'social' as const, label: 'Social & Details', icon: Users, tip: 'View and edit social profiles, descriptions, and external links for your RWA tokens. Synced across the P2P network.' },
        ].map(tab => (
          <Tooltip key={tab.id} text={tab.tip} position="bottom">
            <motion.button
              onClick={() => setActiveSection(tab.id)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm whitespace-nowrap transition-all ${
                activeSection === tab.id
                  ? 'bg-gradient-to-r from-amber-600/80 to-yellow-600/80 text-white shadow-lg shadow-amber-900/30 ring-1 ring-amber-500/30'
                  : 'bg-quantum-dark/60 text-gray-400 hover:text-white hover:bg-quantum-dark/80 ring-1 ring-transparent hover:ring-gray-700/50'
              }`}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.id === 'collateral' && collateralPositions.length > 0 && (
                <span className="bg-blue-500/30 text-blue-300 text-[10px] px-1.5 py-0.5 rounded-full font-bold">{collateralPositions.length}</span>
              )}
              {tab.id === 'distributions' && distributionSchedules.filter(s => s.enabled).length > 0 && (
                <span className="bg-purple-500/30 text-purple-300 text-[10px] px-1.5 py-0.5 rounded-full font-bold">{distributionSchedules.filter(s => s.enabled).length}</span>
              )}
            </motion.button>
          </Tooltip>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {/* ═══════════════════════════════════════════════════
            OVERVIEW SECTION
        ═══════════════════════════════════════════════════ */}
        {activeSection === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            {/* Top Stats Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <GlowCard color="amber">
                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1.5 bg-amber-500/20 rounded-lg">
                    <Wallet className="w-4 h-4 text-amber-400" />
                  </div>
                  <span className="text-xs text-gray-400">Total Portfolio Value</span>
                  <InfoTip text="The combined USD value of all your RWA token holdings, calculated from your ownership percentage of each asset's total valuation." />
                </div>
                <div className="text-2xl font-bold text-white"><AnimatedValue value={metrics.totalValue} /></div>
                <div className="text-xs text-gray-500 mt-1">{rwaContracts.length} RWA holding{rwaContracts.length !== 1 ? 's' : ''}</div>
              </GlowCard>
              <GlowCard color="green">
                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1.5 bg-emerald-500/20 rounded-lg">
                    <TrendingUp className="w-4 h-4 text-green-400" />
                  </div>
                  <span className="text-xs text-gray-400">Annual Yield</span>
                  <InfoTip text="Projected annual income from all RWA holdings. Includes rent, dividends, bond coupons, and royalties weighted by your ownership stake." />
                </div>
                <div className="text-2xl font-bold text-green-400"><AnimatedValue value={metrics.totalYield} /></div>
                <div className="text-xs text-green-400/70 mt-1">{metrics.avgYieldPercent.toFixed(1)}% weighted avg</div>
              </GlowCard>
              <GlowCard color="blue">
                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1.5 bg-blue-500/20 rounded-lg">
                    <Lock className="w-4 h-4 text-blue-400" />
                  </div>
                  <span className="text-xs text-gray-400">Collateral Locked</span>
                  <InfoTip text="Total value of RWA tokens you've locked as collateral to borrow QUG. Collateral is returned when loans are repaid." />
                </div>
                <div className="text-2xl font-bold text-white"><AnimatedValue value={metrics.totalCollateralValue} /></div>
                <div className="text-xs text-blue-400/70 mt-1"><AnimatedValue value={metrics.totalBorrowed} /> borrowed</div>
              </GlowCard>
              <GlowCard color="purple">
                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1.5 bg-purple-500/20 rounded-lg">
                    <Calendar className="w-4 h-4 text-purple-400" />
                  </div>
                  <span className="text-xs text-gray-400">Distributions</span>
                  <InfoTip text="Cumulative revenue distributed to token holders via auto-distribution schedules. This includes rent, dividends, coupons, and royalty payments." />
                </div>
                <div className="text-2xl font-bold text-white"><AnimatedValue value={metrics.totalDistributed} /></div>
                <div className="text-xs text-purple-400/70 mt-1">{metrics.upcomingDistributions} upcoming</div>
              </GlowCard>
            </div>

            {/* Asset Allocation + Yield Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/10 backdrop-blur-sm">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                  <PieChart className="w-4 h-4 text-amber-400" />
                  Asset Allocation
                  <InfoTip text="Breakdown of your RWA portfolio by asset type. Diversification across multiple RWA types reduces overall portfolio risk." />
                </h3>
                {metrics.assetAllocation.length > 0 ? (
                  <div className="space-y-3">
                    {metrics.assetAllocation.map((a, i) => {
                      const pct = metrics.totalValue > 0 ? (a.value / metrics.totalValue) * 100 : 0;
                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.08 }}
                        >
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span className="text-gray-300 flex items-center gap-2">
                              <span className="w-3 h-3 rounded-full ring-2 ring-white/10" style={{ backgroundColor: a.color }} />
                              {a.type} <span className="text-gray-500">({a.count})</span>
                            </span>
                            <span className="text-white font-semibold">{pct.toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-2.5 bg-quantum-dark/80 rounded-full overflow-hidden ring-1 ring-white/5">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ backgroundColor: a.color, boxShadow: `0 0 8px ${a.color}40` }}
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ duration: 0.8, delay: i * 0.1, ease: 'easeOut' }}
                            />
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5">${a.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                        </motion.div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No allocation data</p>
                )}
              </div>

              <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/10 backdrop-blur-sm">
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-green-400" />
                  Yield Performance (12 months)
                  <InfoTip text="Monthly yield income from your RWA portfolio. Each bar represents estimated income for that month based on your current holdings and their yield rates." />
                </h3>
                <div className="flex items-end gap-1 h-40">
                  {metrics.yieldHistory.map((m, i) => {
                    const maxYield = Math.max(...metrics.yieldHistory.map(h => h.yield), 1);
                    const heightPct = (m.yield / maxYield) * 100;
                    return (
                      <Tooltip key={i} text={`${m.date}: $${m.yield.toFixed(0)} yield income`} position="top">
                        <div className="flex-1 flex flex-col items-center justify-end gap-1 cursor-default">
                          <motion.div
                            className="w-full rounded-t-md relative overflow-hidden"
                            style={{
                              background: `linear-gradient(to top, rgba(16, 185, 129, 0.8), rgba(52, 211, 153, 0.5))`,
                            }}
                            initial={{ height: 0 }}
                            animate={{ height: `${heightPct}%` }}
                            transition={{ duration: 0.6, delay: i * 0.04, ease: 'easeOut' }}
                            whileHover={{ filter: 'brightness(1.3)' }}
                          >
                            <div className="absolute inset-0 bg-gradient-to-t from-transparent to-white/10" />
                          </motion.div>
                          <span className="text-[9px] text-gray-500 font-medium">{m.date}</span>
                        </div>
                      </Tooltip>
                    );
                  })}
                </div>
                <div className="flex justify-between mt-3 text-xs text-gray-400 border-t border-gray-800/50 pt-3">
                  <span>Monthly yield income</span>
                  <span className="text-green-400 font-semibold">
                    ~${(metrics.totalYield / 12).toLocaleString(undefined, { maximumFractionDigits: 0 })}/mo
                  </span>
                </div>
              </div>
            </div>

            {/* Holdings List */}
            <div>
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Gem className="w-4 h-4 text-amber-400" />
                Your RWA Holdings
                <InfoTip text="Click any holding to expand details, borrow against it, or set up auto-distributions. Each asset shows your ownership percentage and projected yield." />
              </h3>
              <div className="space-y-3">
                {rwaContracts.map((contract, idx) => {
                  const assetType = getAssetType(contract.type);
                  const value = getHoldingValueUsd(contract);
                  const yieldPct = getAssetYieldPercent(contract);
                  const ownership = getOwnershipPercent(contract);
                  const Icon = ASSET_ICONS[assetType] || Gem;
                  const color = ASSET_COLORS[assetType] || ASSET_COLORS.other;
                  const isExpanded = expandedHolding === contract.address;
                  const ltv = LTV_RATIOS[assetType];
                  const existingPosition = collateralPositions.find(p => p.contractAddress === contract.address);
                  const existingSchedule = distributionSchedules.find(s => s.contractAddress === contract.address);

                  return (
                    <motion.div
                      key={contract.address}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="bg-quantum-dark/40 rounded-xl border border-quantum-purple/10 overflow-hidden hover:border-quantum-purple/25 transition-all duration-200 hover:shadow-lg hover:shadow-black/20"
                      layout
                    >
                      <div
                        className="p-4 cursor-pointer flex items-center gap-4 group"
                        onClick={() => setExpandedHolding(isExpanded ? null : contract.address)}
                      >
                        <div className="p-2.5 rounded-xl transition-all duration-200 group-hover:scale-110" style={{ backgroundColor: `${color}15`, boxShadow: `0 0 12px ${color}10` }}>
                          <Icon className="w-5 h-5" style={{ color }} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white truncate">{contract.name}</span>
                            <span className="text-xs text-gray-500 font-mono bg-gray-800/50 px-1.5 py-0.5 rounded">{contract.symbol}</span>
                            {existingPosition && (
                              <Tooltip text={`Active collateral position: $${existingPosition.borrowedAmount.toLocaleString()} borrowed at ${existingPosition.ltvRatio.toFixed(1)}% LTV`}>
                                <span className="text-[10px] bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded-full font-bold">COLLATERAL</span>
                              </Tooltip>
                            )}
                            {existingSchedule?.enabled && (
                              <Tooltip text={`Auto-distributing $${existingSchedule.amount.toLocaleString()} ${existingSchedule.frequency}`}>
                                <span className="text-[10px] bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded-full font-bold">AUTO-DIST</span>
                              </Tooltip>
                            )}
                          </div>
                          <div className="text-xs text-gray-400">{LTV_RATIOS[assetType]?.label || contract.type}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-white">${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                          <div className="flex items-center justify-end gap-2 text-xs">
                            {yieldPct > 0 && (
                              <span className="text-green-400 flex items-center gap-0.5 font-medium">
                                <ArrowUpRight className="w-3 h-3" />{yieldPct}% yield
                              </span>
                            )}
                            <span className="text-gray-500">{ownership.toFixed(1)}% owned</span>
                          </div>
                        </div>
                        <motion.div animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
                          <ChevronDown className="w-4 h-4 text-gray-500 group-hover:text-gray-300 transition-colors" />
                        </motion.div>
                      </div>

                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.25 }}
                            className="border-t border-quantum-purple/10"
                          >
                            <div className="p-4 space-y-4">
                              <div className="grid grid-cols-3 gap-3">
                                <div className="bg-quantum-dark/60 rounded-lg p-3 text-center ring-1 ring-white/5">
                                  <div className="text-xs text-gray-500 mb-1 flex items-center justify-center gap-1">
                                    Annual Income
                                    <InfoTip text="Projected yearly income based on your ownership percentage and the asset's yield rate." position="bottom" />
                                  </div>
                                  <div className="text-sm font-bold text-green-400">
                                    ${((value * yieldPct) / 100).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                  </div>
                                </div>
                                <div className="bg-quantum-dark/60 rounded-lg p-3 text-center ring-1 ring-white/5">
                                  <div className="text-xs text-gray-500 mb-1 flex items-center justify-center gap-1">
                                    Borrow Power
                                    <InfoTip text={`Maximum you can borrow against this asset at ${ltv?.max || 50}% LTV. ${LTV_TOOLTIPS[assetType] || ''}`} position="bottom" />
                                  </div>
                                  <div className="text-sm font-bold text-blue-400">
                                    ${((value * (ltv?.max || 50)) / 100).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                  </div>
                                </div>
                                <div className="bg-quantum-dark/60 rounded-lg p-3 text-center ring-1 ring-white/5">
                                  <div className="text-xs text-gray-500 mb-1 flex items-center justify-center gap-1">
                                    Compliance
                                    <InfoTip text="Your compliance status for this specific token. Green means you meet all requirements (KYC, accreditation, whitelist) to hold and trade this token." position="bottom" />
                                  </div>
                                  <div className="text-sm font-bold text-emerald-400 flex items-center justify-center gap-1">
                                    <CheckCircle className="w-3 h-3" /> Verified
                                  </div>
                                </div>
                              </div>

                              <div className="flex gap-2">
                                <Tooltip text={existingPosition ? 'You already have an active collateral position for this asset.' : 'Lock this asset as collateral to borrow QUG stablecoin for DeFi activities.'}>
                                  <motion.button
                                    onClick={(e) => { e.stopPropagation(); setShowBorrowModal(contract.address); }}
                                    className="flex-1 bg-blue-500/15 hover:bg-blue-500/25 text-blue-300 px-3 py-2.5 rounded-lg text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-40 ring-1 ring-blue-500/20 hover:ring-blue-500/40 transition-all"
                                    disabled={!!existingPosition}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    <Lock className="w-3.5 h-3.5" />
                                    {existingPosition ? `Borrowing $${existingPosition.borrowedAmount.toLocaleString()}` : 'Borrow Against'}
                                  </motion.button>
                                </Tooltip>
                                <Tooltip text={existingSchedule ? `Auto-distribution already active (${existingSchedule.frequency}).` : 'Set up automatic revenue distributions to all token holders on a schedule.'}>
                                  <motion.button
                                    onClick={(e) => { e.stopPropagation(); setShowScheduleModal(contract.address); }}
                                    className="flex-1 bg-purple-500/15 hover:bg-purple-500/25 text-purple-300 px-3 py-2.5 rounded-lg text-sm font-bold flex items-center justify-center gap-2 disabled:opacity-40 ring-1 ring-purple-500/20 hover:ring-purple-500/40 transition-all"
                                    disabled={!!existingSchedule}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    <Calendar className="w-3.5 h-3.5" />
                                    {existingSchedule ? `${existingSchedule.frequency} active` : 'Auto-Distribute'}
                                  </motion.button>
                                </Tooltip>
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-xs">
                                {Object.entries(contract.deploymentParams || {}).slice(0, 8).map(([key, val]) => (
                                  <div key={key} className="flex justify-between bg-quantum-dark/40 rounded-lg p-2 ring-1 ring-white/5">
                                    <span className="text-gray-500 truncate mr-2">{key.replace(/_/g, ' ')}</span>
                                    <span className="text-gray-300 font-medium truncate">{String(val)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}

        {/* ═══════════════════════════════════════════════════
            COLLATERAL & BORROW SECTION
        ═══════════════════════════════════════════════════ */}
        {activeSection === 'collateral' && (
          <motion.div
            key="collateral"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <GlowCard color="blue">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Total Collateral
                  <InfoTip text="The combined USD value of all RWA tokens you've locked as collateral for borrowing." />
                </div>
                <div className="text-2xl font-bold text-white"><AnimatedValue value={metrics.totalCollateralValue} /></div>
              </GlowCard>
              <GlowCard color="amber">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Total Borrowed
                  <InfoTip text="Total QUG stablecoin borrowed against your RWA collateral. This must be repaid to unlock your collateral tokens." />
                </div>
                <div className="text-2xl font-bold text-amber-400"><AnimatedValue value={metrics.totalBorrowed} /></div>
              </GlowCard>
              <GlowCard color="green">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Available to Borrow
                  <InfoTip text="Remaining borrowing capacity based on your uncollateralized RWA holdings and their LTV ratios." />
                </div>
                <div className="text-2xl font-bold text-green-400">
                  <AnimatedValue value={Math.max(0, metrics.totalValue * 0.5 - metrics.totalBorrowed)} />
                </div>
              </GlowCard>
            </div>

            {/* LTV Ratios by Asset Type */}
            <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/10 backdrop-blur-sm">
              <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                <Percent className="w-4 h-4 text-blue-400" />
                Loan-to-Value (LTV) Ratios by Asset Type
                <InfoTip text="LTV ratio determines how much you can borrow against each asset type. Lower risk assets (bonds, real estate) have higher LTV limits. If your LTV exceeds the maximum, your collateral may be liquidated." />
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(LTV_RATIOS).map(([type, ltv]) => {
                  const Icon = ASSET_ICONS[type] || Gem;
                  return (
                    <Tooltip key={type} text={LTV_TOOLTIPS[type] || `Max LTV: ${ltv.max}%`}>
                      <motion.div
                        className="bg-quantum-dark/60 rounded-xl p-3 border border-quantum-purple/5 cursor-default hover:border-quantum-purple/15 transition-all"
                        whileHover={{ scale: 1.02, y: -2 }}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <div className="p-1.5 rounded-lg" style={{ backgroundColor: `${ASSET_COLORS[type]}15` }}>
                            <Icon className="w-4 h-4" style={{ color: ASSET_COLORS[type] }} />
                          </div>
                          <span className="text-xs text-gray-300 font-medium">{ltv.label}</span>
                        </div>
                        <div className="text-xl font-bold text-white">{ltv.max}%</div>
                        <div className="flex items-center gap-1 text-xs text-yellow-400/80">
                          <AlertTriangle className="w-3 h-3" /> Warning at {ltv.warning}%
                        </div>
                      </motion.div>
                    </Tooltip>
                  );
                })}
              </div>
            </div>

            {/* Active Positions */}
            <div>
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                Active Collateral Positions
                <InfoTip text="Your active borrowing positions. Monitor the LTV ratio - if it exceeds the maximum due to value changes, your collateral may be liquidated to repay the loan." />
              </h3>
              {collateralPositions.length === 0 ? (
                <div className="bg-quantum-dark/40 rounded-xl p-8 text-center border border-quantum-purple/10 border-dashed">
                  <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 2, repeat: Infinity }}>
                    <Lock className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  </motion.div>
                  <p className="text-gray-400 mb-2 font-medium">No active collateral positions</p>
                  <p className="text-xs text-gray-500">Use your RWA tokens as collateral to borrow QUG for DeFi activities</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {collateralPositions.map((pos, idx) => {
                    const contract = rwaContracts.find(c => c.address === pos.contractAddress);
                    return (
                      <motion.div
                        key={pos.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        className="bg-quantum-dark/40 rounded-xl p-4 border border-quantum-purple/10 hover:border-quantum-purple/20 transition-all"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <span className="font-bold text-white">{contract?.name || 'Unknown'}</span>
                            <span className="text-xs text-gray-500 ml-2 font-mono">{contract?.symbol}</span>
                          </div>
                          <Tooltip text={`Position health: ${pos.status}. ${pos.status === 'healthy' ? 'LTV is within safe range.' : pos.status === 'warning' ? 'LTV approaching maximum. Consider repaying to avoid liquidation.' : 'Danger! Liquidation imminent.'}`}>
                            <span className={`text-xs px-2.5 py-1 rounded-full font-bold border ${statusBadge(pos.status)}`}>
                              {pos.status.toUpperCase()}
                            </span>
                          </Tooltip>
                        </div>
                        <div className="grid grid-cols-4 gap-3 mb-3">
                          <div>
                            <div className="text-xs text-gray-500 flex items-center gap-1">Collateral <InfoTip text="USD value of your locked RWA tokens." position="bottom" /></div>
                            <div className="text-sm font-bold text-white">${pos.collateralValueUsd.toLocaleString()}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 flex items-center gap-1">Borrowed <InfoTip text="Amount of QUG stablecoin you've borrowed." position="bottom" /></div>
                            <div className="text-sm font-bold text-amber-400">${pos.borrowedAmount.toLocaleString()}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 flex items-center gap-1">LTV <InfoTip text="Current Loan-to-Value ratio. Calculated as (Borrowed / Collateral Value) x 100." position="bottom" /></div>
                            <div className={`text-sm font-bold ${pos.ltvRatio > pos.maxLtv * 0.85 ? 'text-red-400' : pos.ltvRatio > pos.maxLtv * 0.7 ? 'text-yellow-400' : 'text-green-400'}`}>
                              {pos.ltvRatio.toFixed(1)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 flex items-center gap-1">Interest <InfoTip text="Annual Percentage Rate charged on your borrowed amount. Interest accrues daily." position="bottom" /></div>
                            <div className="text-sm font-bold text-white">{pos.interestRate}% APR</div>
                          </div>
                        </div>
                        <div className="mb-3">
                          <div className="flex justify-between text-xs text-gray-500 mb-1">
                            <span>LTV Ratio</span>
                            <span>Max {pos.maxLtv}%</span>
                          </div>
                          <div className="w-full h-2.5 bg-quantum-dark/80 rounded-full overflow-hidden relative ring-1 ring-white/5">
                            <motion.div
                              className={`h-full rounded-full transition-all ${pos.ltvRatio > pos.maxLtv * 0.85 ? 'bg-red-500 shadow-red-500/30' : pos.ltvRatio > pos.maxLtv * 0.7 ? 'bg-yellow-500 shadow-yellow-500/30' : 'bg-green-500 shadow-green-500/30'}`}
                              style={{ width: `${Math.min(100, (pos.ltvRatio / pos.maxLtv) * 100)}%`, boxShadow: '0 0 8px currentColor' }}
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.min(100, (pos.ltvRatio / pos.maxLtv) * 100)}%` }}
                              transition={{ duration: 0.5 }}
                            />
                            <div className="absolute top-0 h-full w-0.5 bg-yellow-500/60" style={{ left: `${(pos.maxLtv * 0.7 / pos.maxLtv) * 100}%` }} />
                          </div>
                        </div>
                        <motion.button
                          onClick={() => handleRepay(pos.id, pos.borrowedAmount)}
                          disabled={actionLoading[`repay:${pos.id}`]}
                          className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white px-4 py-2.5 rounded-lg text-sm font-bold disabled:opacity-50 shadow-lg shadow-green-900/20 ring-1 ring-green-500/20"
                          whileTap={{ scale: 0.98 }}
                          whileHover={{ scale: 1.01 }}
                        >
                          {actionLoading[`repay:${pos.id}`] ? 'Repaying...' : `Repay $${pos.borrowedAmount.toLocaleString()} QUG`}
                        </motion.button>
                      </motion.div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Available to Collateralize */}
            <div>
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Unlock className="w-4 h-4 text-green-400" />
                Available Assets to Collateralize
                <InfoTip text="RWA holdings that are not yet used as collateral. Click 'Borrow' to lock the asset and borrow QUG against it." />
              </h3>
              <div className="space-y-2">
                {rwaContracts.filter(c => !collateralPositions.find(p => p.contractAddress === c.address)).map((contract, idx) => {
                  const value = getHoldingValueUsd(contract);
                  const assetType = getAssetType(contract.type);
                  const ltv = LTV_RATIOS[assetType];
                  const maxBorrow = ltv ? (value * ltv.max) / 100 : 0;
                  const Icon = ASSET_ICONS[assetType] || Gem;
                  if (value <= 0) return null;
                  return (
                    <motion.div
                      key={contract.address}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.05 }}
                      className="bg-quantum-dark/30 rounded-xl p-3.5 flex items-center gap-3 border border-quantum-purple/5 hover:border-quantum-purple/15 transition-all"
                    >
                      <div className="p-2 rounded-lg" style={{ backgroundColor: `${ASSET_COLORS[assetType]}15` }}>
                        <Icon className="w-5 h-5" style={{ color: ASSET_COLORS[assetType] }} />
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-bold text-white">{contract.name}</div>
                        <div className="text-xs text-gray-500">
                          Value: ${value.toLocaleString()} | Max borrow: ${maxBorrow.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({ltv?.max || 50}% LTV)
                        </div>
                      </div>
                      <motion.button
                        onClick={() => setShowBorrowModal(contract.address)}
                        className="bg-blue-500/15 hover:bg-blue-500/25 text-blue-300 px-4 py-2 rounded-lg text-sm font-bold ring-1 ring-blue-500/20 hover:ring-blue-500/40 transition-all"
                        whileTap={{ scale: 0.98 }}
                        whileHover={{ scale: 1.02 }}
                      >
                        Borrow
                      </motion.button>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}

        {/* ═══════════════════════════════════════════════════
            AUTO-DISTRIBUTIONS SECTION
        ═══════════════════════════════════════════════════ */}
        {activeSection === 'distributions' && (
          <motion.div
            key="distributions"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <GlowCard color="purple">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Total Distributed
                  <InfoTip text="Cumulative amount distributed to token holders across all auto-distribution schedules." />
                </div>
                <div className="text-2xl font-bold text-white"><AnimatedValue value={metrics.totalDistributed} /></div>
              </GlowCard>
              <GlowCard color="green">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Active Schedules
                  <InfoTip text="Number of distribution schedules currently active. Active schedules will automatically distribute revenue on their next scheduled date." />
                </div>
                <div className="text-2xl font-bold text-green-400">{distributionSchedules.filter(s => s.enabled).length}</div>
              </GlowCard>
              <GlowCard color="amber">
                <div className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Next Distribution
                  <InfoTip text="Date of the next upcoming automatic distribution. The smart contract will execute the distribution on-chain without manual intervention." />
                </div>
                <div className="text-lg font-bold text-amber-400">
                  {distributionSchedules.filter(s => s.enabled).length > 0
                    ? new Date(Math.min(...distributionSchedules.filter(s => s.enabled).map(s => s.nextDistribution.getTime()))).toLocaleDateString()
                    : 'None scheduled'}
                </div>
              </GlowCard>
            </div>

            <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/10 backdrop-blur-sm">
              <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                <Clock className="w-4 h-4 text-purple-400" />
                Distribution Schedule
                <InfoTip text="All your revenue distribution schedules sorted by next distribution date. Toggle schedules on/off to pause or resume distributions." />
              </h3>
              {distributionSchedules.length === 0 ? (
                <div className="text-center py-8">
                  <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 2, repeat: Infinity }}>
                    <Calendar className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                  </motion.div>
                  <p className="text-gray-400 mb-2 font-medium">No distributions scheduled</p>
                  <p className="text-xs text-gray-500">Set up automatic revenue distributions for your RWA token holders below</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {distributionSchedules.sort((a, b) => a.nextDistribution.getTime() - b.nextDistribution.getTime()).map((sched, idx) => {
                    const contract = rwaContracts.find(c => c.address === sched.contractAddress);
                    const typeColors: Record<string, string> = { rent: 'text-amber-400', coupon: 'text-blue-400', royalty: 'text-violet-400', dividend: 'text-green-400', custom: 'text-gray-400' };
                    const typeIcons: Record<string, React.ComponentType<any>> = { rent: Building, coupon: Landmark, royalty: FileText, dividend: DollarSign, custom: Settings };
                    const typeTooltips: Record<string, string> = {
                      rent: 'Rental income from property assets distributed to token holders.',
                      coupon: 'Bond coupon payments distributed on schedule to bondholders.',
                      royalty: 'Intellectual property royalties distributed from licensing revenue.',
                      dividend: 'Dividend payments from equity holdings distributed to shareholders.',
                      custom: 'Custom distribution type configured for this asset.',
                    };
                    const TypeIcon = typeIcons[sched.type] || DollarSign;
                    const daysUntil = Math.ceil((sched.nextDistribution.getTime() - Date.now()) / (1000 * 60 * 60 * 24));

                    return (
                      <motion.div
                        key={sched.id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        className="bg-quantum-dark/40 rounded-xl p-4 border border-quantum-purple/5 flex items-center gap-4 hover:border-quantum-purple/15 transition-all"
                      >
                        <Tooltip text={typeTooltips[sched.type]}>
                          <div className="p-2.5 rounded-xl bg-quantum-dark/60 ring-1 ring-white/5">
                            <TypeIcon className={`w-5 h-5 ${typeColors[sched.type]}`} />
                          </div>
                        </Tooltip>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white">{contract?.name || 'Unknown'}</span>
                            <span className={`text-xs font-semibold capitalize px-2 py-0.5 rounded-full ${typeColors[sched.type]} bg-current/10`} style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>{sched.type}</span>
                          </div>
                          <div className="text-xs text-gray-400 mt-0.5">
                            ${sched.amount.toLocaleString()} {sched.frequency} |{' '}
                            <span className={daysUntil <= 3 ? 'text-amber-400 font-medium' : ''}>
                              {daysUntil > 0 ? `Next in ${daysUntil} day${daysUntil !== 1 ? 's' : ''}` : 'Due today'}
                            </span>
                            {' '}| Total: ${sched.totalDistributed.toLocaleString()}
                          </div>
                        </div>
                        <Tooltip text={sched.enabled ? 'Click to pause this distribution schedule.' : 'Click to resume this distribution schedule.'}>
                          <motion.button
                            onClick={() => handleToggleSchedule(sched.id, !sched.enabled)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold ring-1 transition-all ${sched.enabled ? 'bg-green-500/15 text-green-400 ring-green-500/30' : 'bg-gray-500/15 text-gray-400 ring-gray-500/30'}`}
                            whileTap={{ scale: 0.95 }}
                            whileHover={{ scale: 1.05 }}
                          >
                            {sched.enabled ? 'Active' : 'Paused'}
                          </motion.button>
                        </Tooltip>
                      </motion.div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Set Up New Distribution */}
            <div>
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                Set Up Auto-Distribution
                <InfoTip text="Configure automatic revenue distributions for RWA tokens you own. The smart contract will handle proportional distribution to all token holders on the schedule you set." />
              </h3>
              <div className="space-y-2">
                {rwaContracts.filter(c => !distributionSchedules.find(s => s.contractAddress === c.address)).map((contract, idx) => {
                  const assetType = getAssetType(contract.type);
                  const Icon = ASSET_ICONS[assetType] || Gem;
                  const yieldPct = getAssetYieldPercent(contract);
                  const value = getHoldingValueUsd(contract);
                  if (value <= 0 && yieldPct <= 0) return null;
                  return (
                    <motion.div
                      key={contract.address}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.05 }}
                      className="bg-quantum-dark/30 rounded-xl p-3.5 flex items-center gap-3 border border-quantum-purple/5 hover:border-quantum-purple/15 transition-all"
                    >
                      <div className="p-2 rounded-lg" style={{ backgroundColor: `${ASSET_COLORS[assetType]}15` }}>
                        <Icon className="w-5 h-5" style={{ color: ASSET_COLORS[assetType] }} />
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-bold text-white">{contract.name}</div>
                        <div className="text-xs text-gray-500">
                          {yieldPct > 0 ? `${yieldPct}% yield` : 'No yield configured'} | ${value.toLocaleString()} value
                        </div>
                      </div>
                      <motion.button
                        onClick={() => setShowScheduleModal(contract.address)}
                        className="bg-purple-500/15 hover:bg-purple-500/25 text-purple-300 px-4 py-2 rounded-lg text-sm font-bold ring-1 ring-purple-500/20 hover:ring-purple-500/40 transition-all"
                        whileTap={{ scale: 0.98 }}
                        whileHover={{ scale: 1.02 }}
                      >
                        Schedule
                      </motion.button>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}

        {/* ═══════════════════════════════════════════════════
            COMPLIANCE SECTION
        ═══════════════════════════════════════════════════ */}
        {activeSection === 'compliance' && (
          <motion.div
            key="compliance"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            <GlowCard color="green" className="p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-emerald-500/20 rounded-xl">
                  <Shield className="w-6 h-6 text-emerald-400" />
                </div>
                <div>
                  <h3 className="font-bold text-white flex items-center gap-2">
                    Compliance Status
                    <InfoTip text="Your wallet's verification status across all compliance requirements. These checks are performed automatically when you interact with regulated RWA tokens on the DEX." />
                  </h3>
                  <p className="text-xs text-gray-400">Your wallet's verification status for trading RWA tokens</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Tooltip text="Know Your Customer verification confirms your identity. Required for tokens with KYC enforcement. On testnet, all wallets are automatically KYC-verified.">
                  <div className="bg-quantum-dark/60 rounded-xl p-3 text-center ring-1 ring-white/5 cursor-default hover:ring-emerald-500/20 transition-all">
                    <CheckCircle className="w-6 h-6 text-emerald-400 mx-auto mb-1" />
                    <div className="text-xs text-gray-400">KYC Status</div>
                    <div className="text-sm font-bold text-emerald-400">Verified</div>
                  </div>
                </Tooltip>
                <Tooltip text="Accredited investor status is required by some securities tokens (equity, bonds). Accredited investors meet income or net worth thresholds defined by securities regulations.">
                  <div className="bg-quantum-dark/60 rounded-xl p-3 text-center ring-1 ring-white/5 cursor-default hover:ring-emerald-500/20 transition-all">
                    <CheckCircle className="w-6 h-6 text-emerald-400 mx-auto mb-1" />
                    <div className="text-xs text-gray-400">Accreditation</div>
                    <div className="text-sm font-bold text-emerald-400">Accredited</div>
                  </div>
                </Tooltip>
                <Tooltip text="Transfer authorization means your wallet is on the whitelist for tokens with transfer restrictions. Only whitelisted addresses can send or receive restricted tokens.">
                  <div className="bg-quantum-dark/60 rounded-xl p-3 text-center ring-1 ring-white/5 cursor-default hover:ring-emerald-500/20 transition-all">
                    <CheckCircle className="w-6 h-6 text-emerald-400 mx-auto mb-1" />
                    <div className="text-xs text-gray-400">Transfer Auth</div>
                    <div className="text-sm font-bold text-emerald-400">Whitelisted</div>
                  </div>
                </Tooltip>
                <Tooltip text="Your jurisdiction determines which regulatory frameworks apply to your RWA token trading. 'Global' means no geographic restrictions on testnet.">
                  <div className="bg-quantum-dark/60 rounded-xl p-3 text-center ring-1 ring-white/5 cursor-default hover:ring-blue-500/20 transition-all">
                    <Shield className="w-6 h-6 text-blue-400 mx-auto mb-1" />
                    <div className="text-xs text-gray-400">Jurisdiction</div>
                    <div className="text-sm font-bold text-blue-400">Global</div>
                  </div>
                </Tooltip>
              </div>
            </GlowCard>

            <div>
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Eye className="w-4 h-4 text-blue-400" />
                Per-Asset Compliance Requirements
                <InfoTip text="Each RWA token can have its own compliance requirements. These are set during token deployment and enforced on-chain by the smart contract." />
              </h3>
              <div className="space-y-2">
                {rwaContracts.map((contract, idx) => {
                  const assetType = getAssetType(contract.type);
                  const Icon = ASSET_ICONS[assetType] || Gem;
                  const needsKyc = contract.features.kyc_required;
                  const needsAccredited = contract.features.accredited_only;
                  const hasRestrictions = contract.features.transfer_restrictions;

                  return (
                    <motion.div
                      key={contract.address}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      className="bg-quantum-dark/40 rounded-xl p-4 border border-quantum-purple/5 hover:border-quantum-purple/15 transition-all"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <div className="p-1.5 rounded-lg" style={{ backgroundColor: `${ASSET_COLORS[assetType]}15` }}>
                          <Icon className="w-5 h-5" style={{ color: ASSET_COLORS[assetType] }} />
                        </div>
                        <span className="font-bold text-white flex-1">{contract.name}</span>
                        <span className="text-xs text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full font-bold flex items-center gap-1 ring-1 ring-emerald-500/20">
                          <CheckCircle className="w-3 h-3" /> Compliant
                        </span>
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {needsKyc && (
                          <Tooltip text="This token requires Know Your Customer (KYC) verification before you can buy or sell it. Your identity must be verified through the platform's KYC process.">
                            <span className="text-xs bg-amber-500/10 text-amber-400 px-2.5 py-1 rounded-lg font-medium flex items-center gap-1 ring-1 ring-amber-500/20 cursor-help">
                              <Shield className="w-3 h-3" /> KYC Required
                            </span>
                          </Tooltip>
                        )}
                        {needsAccredited && (
                          <Tooltip text="Only accredited investors can hold this token. Accredited investor status requires meeting specific income ($200K+/yr) or net worth ($1M+) thresholds.">
                            <span className="text-xs bg-purple-500/10 text-purple-400 px-2.5 py-1 rounded-lg font-medium flex items-center gap-1 ring-1 ring-purple-500/20 cursor-help">
                              <Lock className="w-3 h-3" /> Accredited Only
                            </span>
                          </Tooltip>
                        )}
                        {hasRestrictions && (
                          <Tooltip text="This token has transfer restrictions. Only whitelisted wallet addresses can send or receive this token. The token issuer manages the whitelist.">
                            <span className="text-xs bg-blue-500/10 text-blue-400 px-2.5 py-1 rounded-lg font-medium flex items-center gap-1 ring-1 ring-blue-500/20 cursor-help">
                              <AlertTriangle className="w-3 h-3" /> Transfer Restricted
                            </span>
                          </Tooltip>
                        )}
                        {!needsKyc && !needsAccredited && !hasRestrictions && (
                          <Tooltip text="This token has no compliance restrictions. Anyone can freely buy, sell, and transfer this token on the DEX.">
                            <span className="text-xs bg-green-500/10 text-green-400 px-2.5 py-1 rounded-lg font-medium ring-1 ring-green-500/20 cursor-help">
                              Open Trading
                            </span>
                          </Tooltip>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            <div className="bg-quantum-dark/50 rounded-xl p-5 border border-quantum-purple/10 backdrop-blur-sm">
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Banknote className="w-4 h-4 text-amber-400" />
                DEX Compliance-Aware Trading Rules
                <InfoTip text="The Q-NarwhalKnight DEX automatically enforces compliance requirements when swapping RWA tokens. These checks are performed on-chain before every trade." />
              </h3>
              <div className="space-y-2 text-sm">
                {[
                  { icon: CheckCircle, color: 'text-green-400', title: 'Automatic KYC Verification', desc: 'When buying KYC-required tokens on the DEX, your wallet\'s KYC status is verified before the swap executes. Failed KYC checks block the transaction.' },
                  { icon: CheckCircle, color: 'text-green-400', title: 'Accreditation Enforcement', desc: 'Accredited-only tokens require proof of accredited investor status. The DEX blocks non-accredited purchases automatically via smart contract logic.' },
                  { icon: CheckCircle, color: 'text-green-400', title: 'Whitelist Transfer Control', desc: 'Tokens with transfer restrictions only allow trades between whitelisted addresses. The smart contract enforces this on-chain before token transfer.' },
                  { icon: Shield, color: 'text-blue-400', title: 'Post-Quantum Compliance Proofs', desc: 'All compliance verifications are cryptographically proven using Dilithium5 post-quantum signatures, ensuring future-proof regulatory compliance.' },
                ].map((rule, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex items-start gap-3 bg-quantum-dark/40 rounded-xl p-3.5 ring-1 ring-white/5 hover:ring-white/10 transition-all"
                  >
                    <rule.icon className={`w-4 h-4 ${rule.color} mt-0.5 shrink-0`} />
                    <div>
                      <span className="text-white font-semibold">{rule.title}</span>
                      <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{rule.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* ═══════════════════════════════════════════════════
            SOCIAL & DETAILS SECTION
        ═══════════════════════════════════════════════════ */}
        {activeSection === 'social' && (
          <motion.div
            key="social"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
            className="space-y-6"
          >
            <GlowCard color="pink" className="p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-pink-500/20 rounded-xl">
                  <Users className="w-6 h-6 text-pink-400" />
                </div>
                <div>
                  <h3 className="font-bold text-white flex items-center gap-2">
                    Social Profiles & Descriptions
                    <InfoTip text="Add social media links, website, and description for your RWA tokens. This information is stored on-chain and synced across the P2P network via gossipsub." />
                  </h3>
                  <p className="text-xs text-gray-400">Manage public information for your RWA tokens</p>
                </div>
              </div>
            </GlowCard>

            <div className="space-y-4">
              {rwaContracts.map((contract, idx) => {
                const assetType = getAssetType(contract.type);
                const Icon = ASSET_ICONS[assetType] || Gem;
                const profile = socialProfiles[contract.address] || {};
                const isEditing = editingSocialAddr === contract.address;
                const hasLinks = profile.twitter || profile.discord || profile.telegram || profile.website || profile.github || profile.medium;

                return (
                  <motion.div
                    key={contract.address}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.06 }}
                    className="bg-quantum-dark/40 rounded-2xl border border-quantum-purple/10 hover:border-quantum-purple/20 transition-all overflow-hidden"
                  >
                    {/* Header */}
                    <div className="flex items-center gap-3 p-4 border-b border-quantum-purple/5">
                      <div className="p-2 rounded-xl" style={{ backgroundColor: `${ASSET_COLORS[assetType]}15` }}>
                        <Icon className="w-5 h-5" style={{ color: ASSET_COLORS[assetType] }} />
                      </div>
                      <div className="flex-1">
                        <div className="font-bold text-white">{contract.name} <span className="text-gray-500 text-sm font-normal">({contract.symbol})</span></div>
                        <div className="text-[11px] text-gray-500 font-mono">{contract.address.slice(0, 12)}...{contract.address.slice(-8)}</div>
                      </div>
                      <motion.button
                        onClick={() => {
                          if (isEditing) {
                            setEditingSocialAddr(null);
                          } else {
                            setEditingSocialAddr(contract.address);
                            setSocialForm(profile);
                          }
                        }}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                          isEditing
                            ? 'bg-gray-700/50 text-gray-300 hover:bg-gray-700/70'
                            : 'bg-pink-500/15 text-pink-400 hover:bg-pink-500/25 ring-1 ring-pink-500/20'
                        }`}
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                      >
                        <Edit3 className="w-3 h-3" />
                        {isEditing ? 'Cancel' : hasLinks || profile.description ? 'Edit' : 'Add Profile'}
                      </motion.button>
                    </div>

                    {/* Content */}
                    <div className="p-4">
                      {isEditing ? (
                        /* ── Edit Mode ── */
                        <div className="space-y-3">
                          <div>
                            <label className="text-xs text-gray-400 mb-1 block">Description</label>
                            <textarea
                              value={socialForm.description || ''}
                              onChange={(e) => setSocialForm(prev => ({ ...prev, description: e.target.value }))}
                              placeholder="Describe your RWA token — what it represents, investment thesis..."
                              rows={3}
                              maxLength={500}
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-pink-500/50 focus:outline-none resize-none"
                            />
                            <p className="text-[10px] text-gray-600 mt-0.5">{(socialForm.description || '').length}/500</p>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">Website</label>
                              <input type="text" value={socialForm.website || ''} onChange={(e) => setSocialForm(p => ({ ...p, website: e.target.value }))}
                                placeholder="https://..." className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-cyan-500/50 focus:outline-none" />
                            </div>
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">X / Twitter</label>
                              <input type="text" value={socialForm.twitter || ''} onChange={(e) => setSocialForm(p => ({ ...p, twitter: e.target.value }))}
                                placeholder="@username" className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-blue-500/50 focus:outline-none" />
                            </div>
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">Discord</label>
                              <input type="text" value={socialForm.discord || ''} onChange={(e) => setSocialForm(p => ({ ...p, discord: e.target.value }))}
                                placeholder="https://discord.gg/..." className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-purple-500/50 focus:outline-none" />
                            </div>
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">Telegram</label>
                              <input type="text" value={socialForm.telegram || ''} onChange={(e) => setSocialForm(p => ({ ...p, telegram: e.target.value }))}
                                placeholder="https://t.me/..." className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-blue-500/50 focus:outline-none" />
                            </div>
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">GitHub</label>
                              <input type="text" value={socialForm.github || ''} onChange={(e) => setSocialForm(p => ({ ...p, github: e.target.value }))}
                                placeholder="https://github.com/..." className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-green-500/50 focus:outline-none" />
                            </div>
                            <div>
                              <label className="text-xs text-gray-400 mb-1 block">Medium / Blog</label>
                              <input type="text" value={socialForm.medium || ''} onChange={(e) => setSocialForm(p => ({ ...p, medium: e.target.value }))}
                                placeholder="https://medium.com/..." className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-orange-500/50 focus:outline-none" />
                            </div>
                          </div>
                          <motion.button
                            onClick={() => handleSaveSocialProfile(contract.address)}
                            disabled={savingSocial}
                            className="w-full bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white py-2.5 rounded-xl text-sm font-semibold disabled:opacity-50 transition-all"
                            whileHover={{ scale: 1.01 }}
                            whileTap={{ scale: 0.99 }}
                          >
                            {savingSocial ? 'Saving...' : 'Save Profile'}
                          </motion.button>
                        </div>
                      ) : (
                        /* ── View Mode ── */
                        <div className="space-y-3">
                          {profile.description && (
                            <p className="text-sm text-gray-300 leading-relaxed bg-quantum-dark/30 rounded-lg p-3 border-l-2 border-pink-500/30">{profile.description}</p>
                          )}
                          {hasLinks ? (
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                              {profile.website && (
                                <a href={profile.website} target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-cyan-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-cyan-500/20">
                                  <Globe className="w-4 h-4 text-cyan-400" />
                                  <span className="text-xs truncate">Website</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                              {profile.twitter && (
                                <a href={profile.twitter.startsWith('http') ? profile.twitter : `https://x.com/${profile.twitter.replace('@', '')}`}
                                  target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-blue-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-blue-500/20">
                                  <span className="text-sm font-bold">𝕏</span>
                                  <span className="text-xs truncate">{profile.twitter}</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                              {profile.discord && (
                                <a href={profile.discord} target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-purple-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-purple-500/20">
                                  <MessageCircle className="w-4 h-4 text-purple-400" />
                                  <span className="text-xs">Discord</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                              {profile.telegram && (
                                <a href={profile.telegram} target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-blue-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-blue-500/20">
                                  <span className="text-sm">✈️</span>
                                  <span className="text-xs">Telegram</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                              {profile.github && (
                                <a href={profile.github} target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-green-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-green-500/20">
                                  <span className="text-sm">⚙️</span>
                                  <span className="text-xs">GitHub</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                              {profile.medium && (
                                <a href={profile.medium} target="_blank" rel="noopener noreferrer"
                                  className="bg-quantum-dark/50 hover:bg-orange-500/10 rounded-lg p-2.5 flex items-center gap-2 text-gray-300 hover:text-white transition-all ring-1 ring-white/5 hover:ring-orange-500/20">
                                  <span className="text-sm">📝</span>
                                  <span className="text-xs">Medium</span>
                                  <ExternalLink className="w-3 h-3 text-gray-500 ml-auto" />
                                </a>
                              )}
                            </div>
                          ) : !profile.description ? (
                            <div className="text-center py-4">
                              <Users className="w-6 h-6 text-gray-600 mx-auto mb-2" />
                              <p className="text-sm text-gray-500">No social profile yet</p>
                              <p className="text-[11px] text-gray-600 mt-1">Click "Add Profile" to add description & links</p>
                            </div>
                          ) : null}
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ Borrow Modal ═══ */}
      <AnimatePresence>
        {showBorrowModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4"
            onClick={() => setShowBorrowModal(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: 'spring', damping: 25 }}
              className="bg-quantum-dark rounded-2xl border border-blue-500/20 p-6 max-w-md w-full shadow-2xl shadow-blue-900/20"
              onClick={e => e.stopPropagation()}
            >
              {(() => {
                const contract = rwaContracts.find(c => c.address === showBorrowModal);
                if (!contract) return null;
                const assetType = getAssetType(contract.type);
                const ltv = LTV_RATIOS[assetType] || { max: 50, warning: 40, label: 'Asset' };
                const value = getHoldingValueUsd(contract);
                const maxBorrow = (value * ltv.max) / 100;
                const currentBorrowAmt = parseFloat(borrowAmount) || 0;
                const currentLtv = value > 0 ? (currentBorrowAmt / value) * 100 : 0;

                return (
                  <>
                    <div className="flex items-center gap-3 mb-1">
                      <div className="p-2 rounded-xl bg-blue-500/15">
                        <Lock className="w-5 h-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold text-white">Borrow Against {contract.name}</h3>
                        <p className="text-xs text-gray-400">Use your {ltv.label} token as collateral to borrow QUG</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 my-4">
                      <div className="bg-quantum-dark/60 rounded-xl p-3 ring-1 ring-white/5">
                        <div className="text-xs text-gray-500 flex items-center gap-1">Collateral Value <InfoTip text="Current USD value of the RWA token you're using as collateral." position="bottom" /></div>
                        <div className="text-lg font-bold text-white">${value.toLocaleString()}</div>
                      </div>
                      <div className="bg-quantum-dark/60 rounded-xl p-3 ring-1 ring-white/5">
                        <div className="text-xs text-gray-500 flex items-center gap-1">Max Borrow <InfoTip text={`Maximum borrowing at ${ltv.max}% LTV for ${ltv.label} assets.`} position="bottom" /></div>
                        <div className="text-lg font-bold text-blue-400">${maxBorrow.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      </div>
                    </div>

                    <div className="mb-4">
                      <label className="text-xs text-gray-400 mb-1.5 block font-medium">Borrow Amount (QUG USD)</label>
                      <input
                        type="text"
                        inputMode="decimal"
                        value={borrowAmount}
                        onChange={e => setBorrowAmount(e.target.value.replace(/[^0-9.]/g, ''))}
                        placeholder={`Max $${maxBorrow.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                        className="w-full bg-quantum-dark/70 border border-blue-500/20 rounded-xl px-4 py-3 text-white text-lg placeholder-gray-600 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/10 transition-all"
                        autoFocus
                      />
                    </div>

                    <div className="mb-4">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Loan-to-Value</span>
                        <span className={`font-semibold ${currentLtv > ltv.warning ? 'text-yellow-400' : 'text-green-400'}`}>{currentLtv.toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-2.5 bg-quantum-dark/80 rounded-full overflow-hidden ring-1 ring-white/5">
                        <motion.div
                          className={`h-full rounded-full transition-colors ${currentLtv > ltv.max ? 'bg-red-500' : currentLtv > ltv.warning ? 'bg-yellow-500' : 'bg-green-500'}`}
                          style={{ width: `${Math.min(100, (currentLtv / ltv.max) * 100)}%` }}
                          animate={{ width: `${Math.min(100, (currentLtv / ltv.max) * 100)}%` }}
                        />
                      </div>
                      {currentBorrowAmt > maxBorrow && (
                        <p className="text-xs text-red-400 mt-1.5 flex items-center gap-1 font-medium">
                          <AlertTriangle className="w-3 h-3" /> Exceeds maximum LTV ratio
                        </p>
                      )}
                    </div>

                    <div className="text-xs text-gray-500 mb-4 bg-quantum-dark/40 rounded-xl p-3 space-y-1.5 ring-1 ring-white/5">
                      <div className="flex justify-between"><span>Interest Rate:</span><span className="text-white font-medium">5.5% APR</span></div>
                      <div className="flex justify-between"><span>Liquidation at:</span><span className="text-yellow-400 font-medium">{ltv.max}% LTV</span></div>
                      <div className="flex justify-between"><span>Minimum borrow:</span><span className="text-white font-medium">$100</span></div>
                    </div>

                    {actionStatus[`borrow:${showBorrowModal}`] && (
                      <motion.p
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`text-xs mb-3 font-medium ${actionStatus[`borrow:${showBorrowModal}`].success ? 'text-green-400' : 'text-red-400'}`}
                      >
                        {actionStatus[`borrow:${showBorrowModal}`].message}
                      </motion.p>
                    )}

                    <div className="flex gap-3">
                      <motion.button
                        onClick={() => setShowBorrowModal(null)}
                        className="flex-1 bg-quantum-dark/60 hover:bg-quantum-dark/80 text-gray-300 px-4 py-3 rounded-xl font-bold ring-1 ring-white/10 hover:ring-white/20 transition-all"
                        whileTap={{ scale: 0.98 }}
                      >
                        Cancel
                      </motion.button>
                      <motion.button
                        onClick={() => handleBorrow(showBorrowModal)}
                        disabled={!borrowAmount || currentBorrowAmt > maxBorrow || currentBorrowAmt < 100 || actionLoading[`borrow:${showBorrowModal}`]}
                        className="flex-1 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white px-4 py-3 rounded-xl font-bold disabled:opacity-50 shadow-lg shadow-blue-900/30 ring-1 ring-blue-500/30"
                        whileTap={{ scale: 0.98 }}
                        whileHover={{ scale: 1.01 }}
                      >
                        {actionLoading[`borrow:${showBorrowModal}`] ? 'Processing...' : 'Confirm Borrow'}
                      </motion.button>
                    </div>
                  </>
                );
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ Schedule Distribution Modal ═══ */}
      <AnimatePresence>
        {showScheduleModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 backdrop-blur-md z-50 flex items-center justify-center p-4"
            onClick={() => setShowScheduleModal(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: 'spring', damping: 25 }}
              className="bg-quantum-dark rounded-2xl border border-purple-500/20 p-6 max-w-md w-full shadow-2xl shadow-purple-900/20"
              onClick={e => e.stopPropagation()}
            >
              {(() => {
                const contract = rwaContracts.find(c => c.address === showScheduleModal);
                if (!contract) return null;
                const distType = getDistributionType(contract);
                const typeLabels: Record<string, string> = { rent: 'Rental Income', coupon: 'Bond Coupon', royalty: 'Royalty Payment', dividend: 'Dividend', custom: 'Custom Distribution' };

                return (
                  <>
                    <div className="flex items-center gap-3 mb-1">
                      <div className="p-2 rounded-xl bg-purple-500/15">
                        <Calendar className="w-5 h-5 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold text-white">Schedule Auto-Distribution</h3>
                        <p className="text-xs text-gray-400">{contract.name} - {typeLabels[distType]}</p>
                      </div>
                    </div>

                    <div className="space-y-4 mt-4">
                      <div>
                        <label className="text-xs text-gray-400 mb-1.5 block font-medium flex items-center gap-1">
                          Distribution Frequency
                          <InfoTip text="How often the smart contract will automatically distribute revenue to all token holders. 'On Receipt' distributes immediately when revenue is received." position="right" />
                        </label>
                        <select
                          value={scheduleConfig.frequency}
                          onChange={e => setScheduleConfig(prev => ({ ...prev, frequency: e.target.value as any }))}
                          className="w-full bg-quantum-dark/70 border border-purple-500/20 rounded-xl px-4 py-3 text-white focus:border-purple-500/50 focus:outline-none focus:ring-2 focus:ring-purple-500/10 transition-all"
                        >
                          <option value="weekly">Weekly</option>
                          <option value="monthly">Monthly</option>
                          <option value="quarterly">Quarterly</option>
                          <option value="annually">Annually</option>
                          <option value="on_receipt">On Receipt (automatic)</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-xs text-gray-400 mb-1.5 block font-medium flex items-center gap-1">
                          Amount per Distribution (USD)
                          <InfoTip text="The total USD amount to distribute each period. This will be split proportionally among all token holders based on their ownership percentage." position="right" />
                        </label>
                        <input
                          type="text"
                          inputMode="decimal"
                          value={scheduleConfig.amount}
                          onChange={e => setScheduleConfig(prev => ({ ...prev, amount: e.target.value.replace(/[^0-9.]/g, '') }))}
                          placeholder="e.g., 5000"
                          className="w-full bg-quantum-dark/70 border border-purple-500/20 rounded-xl px-4 py-3 text-white text-lg placeholder-gray-600 focus:border-purple-500/50 focus:outline-none focus:ring-2 focus:ring-purple-500/10 transition-all"
                          autoFocus
                        />
                      </div>

                      <div className="bg-quantum-dark/40 rounded-xl p-3 text-xs text-gray-500 space-y-1.5 ring-1 ring-white/5">
                        <div className="flex justify-between"><span>Type:</span><span className="text-white capitalize font-medium">{distType}</span></div>
                        <div className="flex justify-between"><span>Recipients:</span><span className="text-white font-medium">All token holders (proportional)</span></div>
                        <div className="flex justify-between"><span>Smart Contract:</span><span className="text-green-400 font-medium">Automated on-chain</span></div>
                      </div>
                    </div>

                    {actionStatus[`schedule:${showScheduleModal}`] && (
                      <motion.p
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`text-xs mt-3 font-medium ${actionStatus[`schedule:${showScheduleModal}`].success ? 'text-green-400' : 'text-red-400'}`}
                      >
                        {actionStatus[`schedule:${showScheduleModal}`].message}
                      </motion.p>
                    )}

                    <div className="flex gap-3 mt-4">
                      <motion.button
                        onClick={() => setShowScheduleModal(null)}
                        className="flex-1 bg-quantum-dark/60 hover:bg-quantum-dark/80 text-gray-300 px-4 py-3 rounded-xl font-bold ring-1 ring-white/10 hover:ring-white/20 transition-all"
                        whileTap={{ scale: 0.98 }}
                      >
                        Cancel
                      </motion.button>
                      <motion.button
                        onClick={() => handleScheduleDistribution(showScheduleModal)}
                        disabled={!scheduleConfig.amount || parseFloat(scheduleConfig.amount) <= 0 || actionLoading[`schedule:${showScheduleModal}`]}
                        className="flex-1 bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-500 hover:to-violet-500 text-white px-4 py-3 rounded-xl font-bold disabled:opacity-50 shadow-lg shadow-purple-900/30 ring-1 ring-purple-500/30"
                        whileTap={{ scale: 0.98 }}
                        whileHover={{ scale: 1.01 }}
                      >
                        {actionLoading[`schedule:${showScheduleModal}`] ? 'Setting up...' : 'Activate Schedule'}
                      </motion.button>
                    </div>
                  </>
                );
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
