import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, TrendingUp, Users, Wallet, Target, Activity, BarChart3, Waves, Anchor, AlertCircle, Info, RefreshCw, HelpCircle, LineChart, Coins, Flame, Clock, Shield, Zap } from 'lucide-react';

// Big, user-friendly tooltip component - FIXED: stays open when hovering tooltip
const BigTooltip: React.FC<{
  children: React.ReactNode;
  title: string;
  explanation: string;
  example?: string;
  position?: 'auto' | 'top' | 'bottom';
}> = ({ children, title, explanation, example, position = 'auto' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [tooltipStyle, setTooltipStyle] = useState<React.CSSProperties>({});
  const [arrowPosition, setArrowPosition] = useState<'top' | 'bottom'>('bottom');
  const triggerRef = React.useRef<HTMLDivElement>(null);
  const closeTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  // Calculate position when opening
  const handleOpen = () => {
    // Clear any pending close timeout
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }

    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const tooltipWidth = 320;
      const tooltipHeight = 200;

      const showBelow = position === 'bottom' || (position === 'auto' && rect.top < 300);

      let left = rect.left + rect.width / 2 - tooltipWidth / 2;
      if (left < 10) left = 10;
      if (left + tooltipWidth > viewportWidth - 10) left = viewportWidth - tooltipWidth - 10;

      let top: number;
      if (showBelow) {
        top = rect.bottom + 12;
        setArrowPosition('top');
      } else {
        top = rect.top - tooltipHeight - 12;
        if (top < 10) {
          top = rect.bottom + 12;
          setArrowPosition('top');
        } else {
          setArrowPosition('bottom');
        }
      }

      setTooltipStyle({
        position: 'fixed',
        top: `${top}px`,
        left: `${left}px`,
        width: `${tooltipWidth}px`,
      });
    }
    setIsOpen(true);
  };

  // Delayed close to allow moving to tooltip
  const handleMouseLeave = () => {
    closeTimeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 150); // Small delay to allow moving to tooltip
  };

  // Keep open when hovering tooltip
  const handleTooltipMouseEnter = () => {
    if (closeTimeoutRef.current) {
      clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
  };

  // Clean up timeout on unmount
  React.useEffect(() => {
    return () => {
      if (closeTimeoutRef.current) {
        clearTimeout(closeTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div
      ref={triggerRef}
      className="relative cursor-help group inline-block"
      onMouseEnter={handleOpen}
      onMouseLeave={handleMouseLeave}
      onClick={() => isOpen ? setIsOpen(false) : handleOpen()}
    >
      {children}
      <HelpCircle className="absolute -top-1 -right-5 w-4 h-4 text-cyan-400/60 group-hover:text-cyan-400 transition-colors" />
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="p-4 rounded-xl shadow-2xl border border-cyan-500/30"
            onMouseEnter={handleTooltipMouseEnter}
            onMouseLeave={handleMouseLeave}
            style={{
              ...tooltipStyle,
              zIndex: 99999,
              background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.99), rgba(30, 58, 95, 0.99))',
              boxShadow: '0 0 40px rgba(34, 211, 238, 0.4), 0 25px 50px -12px rgba(0, 0, 0, 0.8)',
              backdropFilter: 'blur(8px)',
            }}
          >
            <div className="text-cyan-400 font-semibold text-sm mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              {title}
            </div>
            <p className="text-gray-200 text-sm leading-relaxed mb-2">{explanation}</p>
            {example && (
              <div className="bg-black/40 p-2 rounded-lg border border-cyan-500/20">
                <p className="text-xs text-cyan-300 font-mono">{example}</p>
              </div>
            )}
            {arrowPosition === 'top' ? (
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full">
                <div className="w-0 h-0 border-l-8 border-r-8 border-b-8 border-transparent border-b-cyan-500/50" />
              </div>
            ) : (
              <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full">
                <div className="w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-cyan-500/50" />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

interface FinanceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface KLawParams {
  carrying_capacity: number;
  friction_mu: number;
  flow_sensitivity_lambda: number;
}

interface FlowDensity {
  staking_flow: number;
  defi_flow: number;
  treasury_flow: number;
  unlock_flow: number;
  exchange_flow: number;
  composite_omega: number;
}

interface ThreeLayerAdoption {
  layer1_savings: number;
  layer2_settlement: number;
  layer3_collateral: number;
  composite_adoption: number;
}

interface KristensenRatio {
  current_adoption: number;
  equilibrium_ceiling: number;
  ratio: number;
  health_status: string;
  health_emoji: string;
  health_description: string;
}

interface HolderCohort {
  name: string;
  emoji: string;
  range: string;
  holder_count: number;
  total_balance: number;
  percentage_holders: number;
  percentage_supply: number;
  monitoring_robot: string;
}

interface AdoptionCheckpoint {
  target_year: number;
  predicted_adoption: number;
  predicted_holders: number;
  status: string;
}

interface FinancialIntelligence {
  timestamp: number;
  k_law_params: KLawParams;
  current_flow: FlowDensity;
  three_layer_adoption: ThreeLayerAdoption;
  kristensen_ratio: KristensenRatio;
  critical_flow_density: number;
  flow_to_critical_ratio: number;
  holder_distribution: HolderCohort[];
  gini_coefficient: number;
  checkpoints: AdoptionCheckpoint[];
  total_holders: number;
  total_supply: number;
  circulating_supply: number;
  staking_percentage: number;
}

interface StablecoinPegMechanism {
  peg_mechanism: string;
  min_collateral_ratio: number;
  liquidation_ratio: number;
  liquidation_bonus: number;
  warning_ratio: number;
  circuit_breaker_pct: number;
}

interface StablecoinBacking {
  total_qugusd_supply: number;
  total_qug_collateral: number;
  qug_price_usd: number;
  total_collateral_value_usd: number;
  system_collateral_ratio: number;
  excess_collateral_usd: number;
  active_positions: number;
  last_oracle_update: number;
}

interface StablecoinTransparency {
  timestamp: number;
  peg_mechanism: StablecoinPegMechanism;
  backing: StablecoinBacking;
  system_health: string;
  health_description: string;
  is_fully_backed: boolean;
  backing_ratio: number;
}

const FinanceModal: React.FC<FinanceModalProps> = ({ isOpen, onClose }) => {
  const [data, setData] = useState<FinancialIntelligence | null>(null);
  const [stablecoinData, setStablecoinData] = useState<StablecoinTransparency | null>(null);
  const [stablecoinLoading, setStablecoinLoading] = useState(false);
  const [stablecoinError, setStablecoinError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'adoption' | 'holders' | 'checkpoints' | 'stablecoin' | 'graphs' | 'emission' | 'qcredit'>('overview');
  const [qcreditStatus, setQcreditStatus] = useState<any>(null);
  const [qcreditLoading, setQcreditLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchFinancialData();
      fetchStablecoinData();
      fetchQCreditData();
    }
  }, [isOpen]);

  const fetchQCreditData = async () => {
    setQcreditLoading(true);
    try {
      const response = await fetch('/api/v1/qcredit/status');
      const result = await response.json();
      if (result.success && result.data) {
        setQcreditStatus(result.data);
      }
    } catch (err) {
      console.error('Failed to fetch QCREDIT status:', err);
    } finally {
      setQcreditLoading(false);
    }
  };

  const fetchFinancialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/finance/intelligence');
      const result = await response.json();
      if (result.success && result.data) {
        setData(result.data);
      } else {
        setError('Failed to load financial data');
      }
    } catch (err) {
      console.error('Failed to fetch financial intelligence:', err);
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const fetchStablecoinData = async () => {
    setStablecoinLoading(true);
    setStablecoinError(null);
    try {
      const response = await fetch('/api/v1/stablecoin/transparency');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      if (result.success && result.data) {
        setStablecoinData(result.data);
      } else {
        setStablecoinError('No stablecoin data available');
      }
    } catch (err) {
      console.error('Failed to fetch stablecoin transparency:', err);
      setStablecoinError('Stablecoin API unavailable');
    } finally {
      setStablecoinLoading(false);
    }
  };

  if (!isOpen) return null;

  const formatNumber = (n: number) => {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'Healthy': return 'from-green-500 to-emerald-400';
      case 'Recovering': return 'from-yellow-500 to-amber-400';
      case 'Overheated': return 'from-orange-500 to-red-400';
      case 'Underperforming': return 'from-yellow-600 to-orange-500';
      case 'Critical': return 'from-red-600 to-red-500';
      default: return 'from-blue-500 to-cyan-400';
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'emission', label: 'Emission', icon: Flame },
    { id: 'qcredit', label: 'QCREDIT', icon: Zap },
    { id: 'graphs', label: 'Graphs', icon: LineChart },
    { id: 'adoption', label: 'Adoption', icon: TrendingUp },
    { id: 'holders', label: 'Holders', icon: Users },
    { id: 'stablecoin', label: 'QUGUSD', icon: Anchor },
    { id: 'checkpoints', label: 'Roadmap', icon: Target },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.85)' }}
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={(e) => e.stopPropagation()}
            data-finance-modal="true"
            className="relative w-full max-w-4xl max-h-[70vh] flex flex-col rounded-2xl overflow-hidden finance-modal-content"
            style={{
              background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95))',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              boxShadow: '0 0 60px rgba(59, 130, 246, 0.2)',
            }}
          >
            {/* Header */}
            <div className="flex-shrink-0 flex items-center justify-between p-5 border-b border-white/10">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20">
                  <Waves className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">K-Law Financial Intelligence</h2>
                  <p className="text-sm text-gray-400">Water Robot Adoption Analytics</p>
                </div>
              </div>
              <button onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex-shrink-0 flex border-b border-white/10 px-4 overflow-x-auto">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-5" style={{ minHeight: 0 }}>
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400" />
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <AlertCircle className="w-12 h-12 text-red-400 mb-4" />
                  <p className="text-gray-400">{error}</p>
                  <button
                    onClick={fetchFinancialData}
                    className="mt-4 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors flex items-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Retry
                  </button>
                </div>
              ) : data ? (
                <div className="space-y-6">
                  {/* Overview Tab */}
                  {activeTab === 'overview' && (
                    <>
                      {/* Kristensen Health */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <div className="flex items-center justify-between mb-4">
                          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                            <Target className="w-5 h-5 text-cyan-400" />
                            Kristensen Ratio Health
                          </h3>
                          <span className="text-3xl">{data.kristensen_ratio.health_emoji}</span>
                        </div>

                        <div className="grid grid-cols-3 gap-4 mb-5">
                          <div className="text-center p-4 rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10">
                            <BigTooltip
                              title="Current Adoption"
                              explanation="This shows how many people are actually using QUG right now, as a percentage of all potential users. Think of it like: if 100 people could use QUG, this shows how many actually do."
                              example="5.2% = About 5 out of every 100 potential users are holding QUG"
                            >
                              <p className="text-3xl font-bold k-law-value" style={{ color: '#ffffff' }}>{(data.kristensen_ratio.current_adoption * 100).toFixed(1)}%</p>
                              <p className="text-xs k-law-label" style={{ color: '#9ca3af' }}>Current Adoption (A_t)</p>
                            </BigTooltip>
                          </div>
                          <div className="text-center p-4 rounded-xl bg-gradient-to-br from-cyan-500/10 to-cyan-500/5 border border-cyan-500/20">
                            <BigTooltip
                              title="Equilibrium Ceiling"
                              explanation="This is the 'natural limit' of adoption based on current network activity. It predicts how high adoption can go given how much money is flowing through the system. If adoption exceeds this, it might be unsustainable."
                              example="If equilibrium is 6%, but adoption is 8%, the network might be overheated"
                            >
                              <p className="text-3xl font-bold k-law-value" style={{ color: '#22d3ee' }}>{(data.kristensen_ratio.equilibrium_ceiling * 100).toFixed(2)}%</p>
                              <p className="text-xs k-law-label" style={{ color: '#9ca3af' }}>Equilibrium (A*_t)</p>
                            </BigTooltip>
                          </div>
                          <div className="text-center p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-purple-500/5 border border-purple-500/20">
                            <BigTooltip
                              title="K-Ratio Health Score"
                              explanation="This is like a 'health check' for the network. It compares actual adoption to sustainable adoption. A score of 1.0 means perfectly healthy. Below 0.8 means underperforming (room to grow). Above 1.2 means possibly overheated (hype exceeding fundamentals)."
                              example="0.95 = Healthy | 0.5 = Underperforming | 1.5 = Overheated"
                            >
                              <p className="text-3xl font-bold k-law-value" style={{ color: '#c084fc' }}>{data.kristensen_ratio.ratio.toFixed(2)}</p>
                              <p className="text-xs k-law-label" style={{ color: '#9ca3af' }}>K_t Ratio</p>
                            </BigTooltip>
                          </div>
                        </div>

                        {/* Health Bar */}
                        <div className="relative h-4 bg-white/10 rounded-full overflow-hidden mb-2">
                          <div
                            className={`h-full bg-gradient-to-r ${getHealthColor(data.kristensen_ratio.health_status)} transition-all duration-500`}
                            style={{ width: `${Math.min(data.kristensen_ratio.ratio * 50, 100)}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 mb-3">
                          <span>0 Critical</span>
                          <span>1.0 Healthy</span>
                          <span>2.0+ Overheated</span>
                        </div>
                        <p className="text-sm text-gray-300 bg-black/20 p-3 rounded-lg">{data.kristensen_ratio.health_description}</p>
                      </div>

                      {/* K-Law Formula */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <BarChart3 className="w-5 h-5 text-purple-400" />
                          K-Law Adoption Formula
                        </h3>
                        <div className="text-center py-5 bg-black/30 rounded-xl font-mono text-lg text-cyan-300 border border-cyan-500/20 mb-4">
                          A*_t = K / (1 + μ · e<sup className="text-sm">-λ·Ω_t</sup>)
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="text-center p-3 rounded-lg bg-white/5 border border-white/10">
                            <BigTooltip
                              title="Carrying Capacity (K)"
                              explanation="The maximum possible adoption - like the 'ceiling' the network could theoretically reach. K=1 means 100% of all potential users could adopt. This is the upper limit in the adoption formula."
                              example="K=1 means at maximum, everyone who could use QUG would use it"
                            >
                              <p className="text-lg font-bold k-law-value" style={{ color: '#ffffff' }}>K = {data.k_law_params.carrying_capacity}</p>
                              <p className="text-xs k-law-label" style={{ color: '#6b7280' }}>Carrying Capacity</p>
                            </BigTooltip>
                          </div>
                          <div className="text-center p-3 rounded-lg bg-white/5 border border-white/10">
                            <BigTooltip
                              title="Friction (μ)"
                              explanation="Think of this as 'resistance to growth'. Higher friction means adoption grows more slowly because of barriers like complexity, competition, or lack of awareness. Lower friction = faster potential growth."
                              example="μ=99 is high friction - adoption faces many obstacles"
                            >
                              <p className="text-lg font-bold k-law-value" style={{ color: '#ffffff' }}>μ = {data.k_law_params.friction_mu}</p>
                              <p className="text-xs k-law-label" style={{ color: '#6b7280' }}>Friction Coefficient</p>
                            </BigTooltip>
                          </div>
                          <div className="text-center p-3 rounded-lg bg-white/5 border border-white/10">
                            <BigTooltip
                              title="Flow Sensitivity (λ)"
                              explanation="How much network activity affects the adoption ceiling. Higher λ means the network responds more dramatically to money flowing through it. When more value flows, the ceiling rises faster."
                              example="λ=2 means network flows have a strong effect on adoption potential"
                            >
                              <p className="text-lg font-bold k-law-value" style={{ color: '#ffffff' }}>λ = {data.k_law_params.flow_sensitivity_lambda}</p>
                              <p className="text-xs k-law-label" style={{ color: '#6b7280' }}>Flow Sensitivity</p>
                            </BigTooltip>
                          </div>
                        </div>
                      </div>

                      {/* Flow Density */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Waves className="w-5 h-5 text-blue-400" />
                          Network Flow Density
                          <BigTooltip
                            title="What is Flow Density (Ω)?"
                            explanation="Flow Density measures all the 'useful activity' happening in the network. It combines staking, DeFi usage, treasury activity, and exchange flows into one number. Higher flow = healthier network with more real usage."
                            example="Ω = 0.05 means 5% of tokens are actively being used in productive ways"
                          >
                            <span className="ml-auto text-sm font-mono flow-density-value" style={{ color: '#22d3ee' }}>Ω_t = {data.current_flow.composite_omega.toFixed(4)}</span>
                          </BigTooltip>
                        </h3>
                        {/* Data source legend */}
                        <div className="flex items-center gap-4 mb-3 text-xs">
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400"></span> Real blockchain data</span>
                          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400"></span> Protocol constant</span>
                        </div>
                        <div className="space-y-3">
                          {[
                            { label: 'Staking Flow', value: data.current_flow.staking_flow, color: '#22d3ee', isReal: true,
                              title: 'Staking Flow (Real Data)',
                              tip: 'Calculated from actual wallet balances on the blockchain. Shows what percentage of tokens are held by large wallets (likely staking). Higher = more long-term believers.',
                              example: 'Source: Real wallet_balances from blockchain state' },
                            { label: 'DeFi Flow', value: data.current_flow.defi_flow, color: '#a855f7', isReal: true,
                              title: 'DeFi Activity (Real Data)',
                              tip: 'Calculated from real Total Value Locked (TVL) in liquidity pools divided by circulating supply. Shows actual DeFi participation.',
                              example: 'Source: Real liquidity_pools TVL from blockchain' },
                            { label: 'Treasury Flow', value: data.current_flow.treasury_flow, color: '#22c55e', isReal: false,
                              title: 'Treasury Holdings (Protocol Constant)',
                              tip: 'Fixed protocol allocation of 10% reserved for treasury. This is a constant defined in the protocol, not calculated from live data.',
                              example: 'Source: Protocol constant = 10%' },
                            { label: 'Unlock Flow', value: data.current_flow.unlock_flow, color: '#eab308', isReal: true,
                              title: 'Token Unlocks (Real Data)',
                              tip: 'Calculated from real circulating supply vs total supply. Shows what percentage of tokens are still locked/vesting.',
                              example: 'Source: Real minted_supply / total_supply ratio' },
                            { label: 'Exchange Flow', value: data.current_flow.exchange_flow, color: '#ec4899', isReal: true,
                              title: 'Exchange Activity (Real Data)',
                              tip: 'Estimated from small wallet balances (<10 QUG) as a proxy for exchange hot wallets. Conservative estimate from real wallet data.',
                              example: 'Source: Real small_holder_balance from wallet_balances' },
                          ].map((flow) => (
                            <div key={flow.label} className="flex items-center gap-3">
                              <BigTooltip title={flow.title} explanation={flow.tip} example={flow.example} position="top">
                                <span className="w-28 text-sm text-gray-400 flex items-center gap-1">
                                  <span className={`w-2 h-2 rounded-full ${flow.isReal ? 'bg-green-400' : 'bg-yellow-400'}`}></span>
                                  {flow.label}
                                </span>
                              </BigTooltip>
                              <div className="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{ width: `${flow.value * 100}%`, backgroundColor: flow.color }}
                                />
                              </div>
                              <span className="w-16 text-right text-sm font-medium flow-density-value" style={{ color: '#ffffff' }}>{(flow.value * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                        <div className="mt-4 p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center gap-2">
                          <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                          <p className="text-sm text-cyan-300">
                            Critical threshold Ω<sup>crit</sup> = {data.critical_flow_density.toFixed(2)} | Current = {(data.flow_to_critical_ratio * 100).toFixed(1)}% of critical
                          </p>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Adoption Tab */}
                  {/* ═══════ EMISSION ECONOMICS TAB ═══════ */}
                  {activeTab === 'emission' && (
                    <>
                      {/* Live Emission State */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/20">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Flame className="w-5 h-5 text-amber-400" />
                          Live Emission State
                        </h3>
                        {(() => {
                          const GENESIS_TS = 1771761600;
                          const SECS_PER_ERA = 126_230_400;
                          const nowSec = Math.floor(Date.now() / 1000);
                          const elapsed = Math.max(0, nowSec - GENESIS_TS);
                          const era = Math.floor(elapsed / SECS_PER_ERA);
                          const eraProgress = ((elapsed % SECS_PER_ERA) / SECS_PER_ERA) * 100;
                          const eraAnnual = 2_625_000 / Math.pow(2, era);
                          const eraDaily = eraAnnual / 365.25;
                          const totalTarget = 21_000_000 * (1 - Math.pow(2, -elapsed / SECS_PER_ERA));
                          const pctMined = (totalTarget / 21_000_000) * 100;
                          const inflationRate = (eraAnnual / Math.max(totalTarget, 1)) * 100;
                          const s2f = totalTarget / eraAnnual;
                          const daysToHalving = Math.floor(((era + 1) * SECS_PER_ERA - elapsed) / 86400);
                          const yrsElapsed = elapsed / (365.25 * 86400);
                          return (
                            <div className="space-y-4">
                              <div className="grid grid-cols-4 gap-3">
                                <div className="bg-black/20 rounded-lg p-3 text-center">
                                  <div className="text-2xl font-bold text-amber-300">{era}</div>
                                  <div className="text-xs text-gray-400">Current Era</div>
                                  <div className="text-[10px] text-gray-500">{eraProgress.toFixed(2)}% complete</div>
                                </div>
                                <div className="bg-black/20 rounded-lg p-3 text-center">
                                  <div className="text-2xl font-bold text-white">{eraDaily.toFixed(1)}</div>
                                  <div className="text-xs text-gray-400">QUG/day target</div>
                                  <div className="text-[10px] text-gray-500">{eraAnnual.toLocaleString()} /yr</div>
                                </div>
                                <div className="bg-black/20 rounded-lg p-3 text-center">
                                  <div className="text-2xl font-bold text-purple-400">{s2f.toFixed(1)}</div>
                                  <div className="text-xs text-gray-400">Stock-to-Flow</div>
                                  <div className="text-[10px] text-gray-500">{inflationRate.toFixed(2)}% inflation</div>
                                </div>
                                <div className="bg-black/20 rounded-lg p-3 text-center">
                                  <div className="text-2xl font-bold text-cyan-300">{daysToHalving.toLocaleString()}</div>
                                  <div className="text-xs text-gray-400">Days to Halving</div>
                                  <div className="text-[10px] text-gray-500">Reward halves to {(eraDaily / 2).toFixed(1)}/day</div>
                                </div>
                              </div>

                              {/* Supply Progress */}
                              <div className="bg-black/20 rounded-lg p-3">
                                <div className="flex justify-between text-xs text-gray-400 mb-1">
                                  <span>Estimated Mined: {totalTarget.toLocaleString('en-US', { maximumFractionDigits: 0 })} QUG</span>
                                  <span>{pctMined.toFixed(4)}% of 21M</span>
                                </div>
                                <div className="h-3 bg-gray-700/50 rounded-full overflow-hidden">
                                  <motion.div
                                    className="h-full bg-gradient-to-r from-amber-500 to-orange-400 rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.max(pctMined, 0.5)}%` }}
                                    transition={{ duration: 1.5 }}
                                  />
                                </div>
                                <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                                  <span>0</span>
                                  <span>21,000,000 QUG</span>
                                </div>
                              </div>
                            </div>
                          );
                        })()}
                      </div>

                      {/* 256-Year Supply Curve (animated) */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Coins className="w-5 h-5 text-amber-400" />
                          256-Year Supply Curve
                        </h3>
                        <div className="relative h-64 bg-black/30 rounded-xl p-4 overflow-hidden">
                          <svg viewBox="0 0 500 220" className="w-full h-full">
                            {/* Grid */}
                            {[0, 5.25, 10.5, 15.75, 21].map((val, i) => (
                              <g key={`g-${i}`}>
                                <line x1="45" y1={200 - i * 45} x2="480" y2={200 - i * 45} stroke="rgba(255,255,255,0.07)" />
                                <text x="40" y={204 - i * 45} fill="#6B7280" fontSize="8" textAnchor="end">{val}M</text>
                              </g>
                            ))}
                            {[0, 32, 64, 96, 128, 160, 192, 224, 256].map(yr => (
                              <g key={`x-${yr}`}>
                                <line x1={45 + yr * 1.7} y1="20" x2={45 + yr * 1.7} y2="205" stroke="rgba(255,255,255,0.05)" />
                                <text x={45 + yr * 1.7} y="215" fill="#6B7280" fontSize="7" textAnchor="middle">{yr}yr</text>
                              </g>
                            ))}

                            {/* Area fill */}
                            <motion.path
                              d={(() => {
                                let path = 'M 45,200 ';
                                for (let yr = 0; yr <= 256; yr += 1) {
                                  const supply = 21 * (1 - Math.pow(2, -yr / 4));
                                  path += `L ${(45 + yr * 1.7).toFixed(1)},${(200 - (supply / 21) * 180).toFixed(1)} `;
                                }
                                path += `L ${(45 + 256 * 1.7).toFixed(1)},200 Z`;
                                return path;
                              })()}
                              fill="url(#supplyAreaGrad)"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 0.3 }}
                              transition={{ duration: 1 }}
                            />

                            {/* Supply curve line */}
                            <motion.path
                              d={(() => {
                                const pts: string[] = [];
                                for (let yr = 0; yr <= 256; yr += 1) {
                                  const supply = 21 * (1 - Math.pow(2, -yr / 4));
                                  pts.push(`${yr === 0 ? 'M' : 'L'}${(45 + yr * 1.7).toFixed(1)},${(200 - (supply / 21) * 180).toFixed(1)}`);
                                }
                                return pts.join(' ');
                              })()}
                              fill="none"
                              stroke="#F59E0B"
                              strokeWidth="2"
                              strokeLinecap="round"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 3, ease: 'easeOut' }}
                            />

                            {/* 21M asymptote */}
                            <line x1="45" y1="20" x2="480" y2="20" stroke="#EF4444" strokeWidth="0.8" strokeDasharray="4,3" />
                            <text x="482" y="23" fill="#EF4444" fontSize="8">21M cap</text>

                            {/* Era lines */}
                            {[4, 8, 12, 16].map(yr => (
                              <g key={`era-${yr}`}>
                                <line x1={45 + yr * 1.7} y1="20" x2={45 + yr * 1.7} y2="200" stroke="rgba(139,92,246,0.3)" strokeDasharray="2,3" />
                                <text x={45 + yr * 1.7 + 2} y="30" fill="#A78BFA" fontSize="6">Era {yr / 4}</text>
                              </g>
                            ))}

                            {/* Current position */}
                            {(() => {
                              const elapsed = Math.max(0, Date.now() / 1000 - 1771761600);
                              const yrs = elapsed / (365.25 * 86400);
                              const supply = 21 * (1 - Math.pow(2, -yrs / 4));
                              const cx = 45 + Math.min(yrs, 256) * 1.7;
                              const cy = 200 - (supply / 21) * 180;
                              return (
                                <motion.circle
                                  cx={Math.min(cx, 480)} cy={Math.max(cy, 20)} r="5"
                                  fill="#3B82F6" stroke="#fff" strokeWidth="1"
                                  initial={{ scale: 0 }}
                                  animate={{ scale: [1, 1.4, 1] }}
                                  transition={{ duration: 2, repeat: Infinity }}
                                />
                              );
                            })()}

                            <defs>
                              <linearGradient id="supplyAreaGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.4" />
                                <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
                              </linearGradient>
                            </defs>
                          </svg>
                        </div>
                        <p className="text-xs text-gray-500 mt-2 text-center">C(t) = 21,000,000 × (1 - 2^(-t/4)) | 64 halvings over 256 years</p>
                      </div>

                      {/* Stock-to-Flow & Scarcity */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Shield className="w-5 h-5 text-purple-400" />
                          Stock-to-Flow: Scarcity Timeline
                        </h3>
                        <div className="relative h-56 bg-black/30 rounded-xl p-4 overflow-hidden">
                          <svg viewBox="0 0 500 200" className="w-full h-full">
                            {/* Grid */}
                            {[0, 50, 100, 200, 500].map((val, i) => {
                              const y = 180 - Math.min(val / 500, 1) * 160;
                              return (
                                <g key={`sf-${i}`}>
                                  <line x1="45" y1={y} x2="480" y2={y} stroke="rgba(255,255,255,0.06)" />
                                  <text x="40" y={y + 3} fill="#6B7280" fontSize="7" textAnchor="end">{val}</text>
                                </g>
                              );
                            })}
                            {[0, 10, 20, 30, 40, 50, 60].map(yr => (
                              <text key={`sfx-${yr}`} x={45 + yr * 7.25} y="195" fill="#6B7280" fontSize="7" textAnchor="middle">{yr}yr</text>
                            ))}

                            {/* S2F Curve */}
                            <motion.path
                              d={(() => {
                                const pts: string[] = [];
                                for (let yr = 0.5; yr <= 60; yr += 0.5) {
                                  const supply = 21e6 * (1 - Math.pow(2, -yr / 4));
                                  const era = Math.floor(yr / 4);
                                  const annual = 2625000 / Math.pow(2, era);
                                  const s2f = supply / annual;
                                  const y = 180 - Math.min(s2f / 500, 1) * 160;
                                  const x = 45 + yr * 7.25;
                                  pts.push(`${pts.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
                                }
                                return pts.join(' ');
                              })()}
                              fill="none" stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 2.5, ease: 'easeOut' }}
                            />

                            {/* Bitcoin S2F reference */}
                            {(() => {
                              const btcY = 180 - Math.min(121 / 500, 1) * 160;
                              return (
                                <>
                                  <line x1="45" y1={btcY} x2="480" y2={btcY} stroke="#F7931A" strokeWidth="1" strokeDasharray="6,3" />
                                  <text x="482" y={btcY + 3} fill="#F7931A" fontSize="7">BTC S2F (121)</text>
                                </>
                              );
                            })()}

                            {/* Gold S2F reference */}
                            {(() => {
                              const goldY = 180 - Math.min(59 / 500, 1) * 160;
                              return (
                                <>
                                  <line x1="45" y1={goldY} x2="480" y2={goldY} stroke="#FFD700" strokeWidth="0.8" strokeDasharray="4,4" />
                                  <text x="482" y={goldY + 3} fill="#FFD700" fontSize="7">Gold (59)</text>
                                </>
                              );
                            })()}

                            {/* Current position */}
                            {(() => {
                              const elapsed = Math.max(0, Date.now() / 1000 - 1771761600);
                              const yrs = elapsed / (365.25 * 86400);
                              const supply = 21e6 * (1 - Math.pow(2, -yrs / 4));
                              const era = Math.floor(yrs / 4);
                              const annual = 2625000 / Math.pow(2, era);
                              const s2f = supply / annual;
                              const x = 45 + Math.min(yrs, 60) * 7.25;
                              const y = 180 - Math.min(s2f / 500, 1) * 160;
                              return (
                                <motion.circle
                                  cx={x} cy={y} r="5" fill="#22D3EE" stroke="#fff" strokeWidth="1"
                                  initial={{ scale: 0 }} animate={{ scale: [1, 1.3, 1] }}
                                  transition={{ duration: 2, repeat: Infinity }}
                                />
                              );
                            })()}
                          </svg>
                        </div>
                        <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
                          <div className="bg-purple-500/10 rounded-lg p-2 text-center">
                            <div className="text-purple-300 font-bold">Year 12</div>
                            <div className="text-gray-400">S2F passes Gold (59)</div>
                          </div>
                          <div className="bg-orange-500/10 rounded-lg p-2 text-center">
                            <div className="text-orange-300 font-bold">Year 16</div>
                            <div className="text-gray-400">S2F passes Bitcoin (121)</div>
                          </div>
                          <div className="bg-cyan-500/10 rounded-lg p-2 text-center">
                            <div className="text-cyan-300 font-bold">Year 20</div>
                            <div className="text-gray-400">S2F = 248 (2x BTC)</div>
                          </div>
                        </div>
                      </div>

                      {/* Halving Schedule Table */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Clock className="w-5 h-5 text-cyan-400" />
                          Halving Schedule (First 10 Eras)
                        </h3>
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="text-gray-500 border-b border-gray-700/50">
                              <th className="text-left py-2">Era</th>
                              <th className="text-left py-2">Period</th>
                              <th className="text-right py-2">Annual QUG</th>
                              <th className="text-right py-2">Daily QUG</th>
                              <th className="text-right py-2">Era Total</th>
                              <th className="text-right py-2">Cumulative %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Array.from({ length: 10 }, (_, k) => {
                              const startYr = 2026 + k * 4;
                              const annual = 2625000 / Math.pow(2, k);
                              const daily = annual / 365.25;
                              const eraTotal = annual * 4;
                              const cumPct = (1 - Math.pow(2, -(k + 1))) * 100;
                              const nowEra = Math.floor(Math.max(0, Date.now() / 1000 - 1771761600) / 126230400);
                              return (
                                <tr key={k} className={`border-b border-gray-800/30 ${k === nowEra ? 'bg-amber-500/10 text-amber-200' : 'text-gray-300'}`}>
                                  <td className="py-2 font-mono">{k}{k === nowEra ? ' ◀' : ''}</td>
                                  <td className="py-2">{startYr}–{startYr + 4}</td>
                                  <td className="py-2 text-right font-mono">{annual >= 1000 ? annual.toLocaleString() : annual.toFixed(2)}</td>
                                  <td className="py-2 text-right font-mono">{daily >= 1 ? daily.toFixed(1) : daily.toFixed(4)}</td>
                                  <td className="py-2 text-right font-mono">{eraTotal >= 1000 ? eraTotal.toLocaleString() : eraTotal.toFixed(2)}</td>
                                  <td className="py-2 text-right font-mono">{cumPct.toFixed(3)}%</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        <div className="mt-3 text-xs text-gray-500 text-center">
                          64 eras × 4 years = 256 years total emission | Sum → 21,000,000 QUG (geometric series proof)
                        </div>
                      </div>

                      {/* Adaptive Reward Invariance */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-cyan-500/10 to-green-500/10 border border-cyan-500/20">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <Activity className="w-5 h-5 text-cyan-400" />
                          Adaptive Reward: Rate-Independent Emission
                        </h3>
                        <p className="text-sm text-gray-400 mb-3">
                          Unlike Bitcoin (fixed 10-min blocks), QUG rewards adapt inversely to throughput.
                          Whether the network produces 1 or 10,000 blocks/sec, annual emission stays constant.
                        </p>
                        <div className="bg-black/20 rounded-lg p-3">
                          <div className="text-xs text-gray-500 mb-2 text-center font-mono">
                            R(λ) = Annual_Target / (λ × 31,557,600) | R × λ × T_year = Annual_Target ∀ λ
                          </div>
                          <div className="grid grid-cols-6 gap-2">
                            {[0.1, 1, 5, 10, 100, 1000].map(rate => {
                              const reward = 2625000 / (rate * 31557600);
                              return (
                                <div key={rate} className="text-center bg-black/20 rounded p-2">
                                  <div className="text-[10px] text-gray-500">{rate >= 1000 ? '1K' : rate} bps</div>
                                  <div className="text-xs font-mono text-amber-300 font-bold">{reward >= 0.001 ? reward.toFixed(4) : reward.toExponential(1)}</div>
                                  <div className="text-[9px] text-green-500/80 font-mono">2.625M/yr</div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      {/* Decentralized Verification */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <Shield className="w-5 h-5 text-green-400" />
                          Decentralized Verification
                        </h3>
                        <div className="space-y-2 text-sm text-gray-300">
                          <div className="flex items-start gap-2">
                            <span className="text-green-400 mt-0.5">1.</span>
                            <span><strong className="text-white">Block Producer</strong> computes reward from measured block rate + era schedule using pure u128 integer arithmetic</span>
                          </div>
                          <div className="flex items-start gap-2">
                            <span className="text-green-400 mt-0.5">2.</span>
                            <span><strong className="text-white">Every Node</strong> independently verifies coinbase amount against the same formula — rejects invalid rewards</span>
                          </div>
                          <div className="flex items-start gap-2">
                            <span className="text-green-400 mt-0.5">3.</span>
                            <span><strong className="text-white">Error Correction</strong> compares cumulative emission to target C*(t), applies smoothed correction factor (α = 0.15)</span>
                          </div>
                          <div className="flex items-start gap-2">
                            <span className="text-green-400 mt-0.5">4.</span>
                            <span><strong className="text-white">Hard Supply Cap</strong>: u128 check ensures total emission never exceeds 21,000,000.000000 QUG</span>
                          </div>
                        </div>
                        <div className="mt-3 bg-black/20 rounded-lg p-2 text-xs text-gray-500 font-mono text-center">
                          No trusted oracle. No coordinator. Pure math from genesis timestamp (1771761600).
                        </div>
                      </div>

                      {/* Attosecond Opto-Physics: Emission Timescale Foundations */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-violet-500/10 to-fuchsia-500/10 border border-violet-500/20">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <Zap className="w-5 h-5 text-violet-400" />
                          Attosecond Opto-Physics: Emission Timescale Foundations
                        </h3>
                        <p className="text-sm text-gray-400 mb-3">
                          QUG emission draws from the mathematics of ultrafast laser physics. Just as attosecond pulses
                          (10<sup>-18</sup>s) resolve electron dynamics in real-time, the emission controller resolves
                          economic dynamics at sub-block granularity using analogous time-energy uncertainty principles.
                        </p>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                          <div className="bg-black/20 rounded-lg p-3">
                            <div className="text-xs text-violet-400 font-semibold mb-2">Pulse-Train Emission Model</div>
                            <p className="text-[11px] text-gray-400 mb-2">
                              Each block reward is an "emission pulse" — a discrete energy packet analogous to an
                              attosecond XUV pulse in a high-harmonic generation (HHG) laser system:
                            </p>
                            <div className="text-xs font-mono text-violet-300 bg-black/30 rounded p-2 text-center">
                              E<sub>pulse</sub>(n) = A<sub>k</sub> / (lambda * T) * rect(t - n*tau)
                            </div>
                            <p className="text-[10px] text-gray-500 mt-1">
                              Where each block at interval tau carries exactly the energy (reward) needed to maintain
                              the target annual flux, independent of repetition rate lambda.
                            </p>
                          </div>

                          <div className="bg-black/20 rounded-lg p-3">
                            <div className="text-xs text-fuchsia-400 font-semibold mb-2">Time-Energy Uncertainty Bound</div>
                            <p className="text-[11px] text-gray-400 mb-2">
                              The emission controller's PI correction factor is bounded by a Heisenberg-inspired
                              time-energy relation — faster measurement means larger reward uncertainty:
                            </p>
                            <div className="text-xs font-mono text-fuchsia-300 bg-black/30 rounded p-2 text-center">
                              Delta_R * Delta_t &ge; hbar_econ = A<sub>k</sub> / (2*pi * N<sub>blocks/year</sub>)
                            </div>
                            <p className="text-[10px] text-gray-500 mt-1">
                              This prevents the correction factor from oscillating — the economic "uncertainty principle"
                              that stabilizes emission even under adversarial block rate manipulation.
                            </p>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                          <div className="bg-black/20 rounded-lg p-3">
                            <div className="text-xs text-cyan-400 font-semibold mb-2">Chirped Halving Envelope</div>
                            <p className="text-[11px] text-gray-400 mb-2">
                              The 64-era halving schedule forms a chirped envelope function, analogous to chirped-pulse
                              amplification (CPA) in femtosecond lasers. Early eras carry high energy (2.625M QUG/yr),
                              with exponential decay:
                            </p>
                            <div className="text-xs font-mono text-cyan-300 bg-black/30 rounded p-2 text-center">
                              A(t) = A<sub>0</sub> * exp(-t * ln2 / T<sub>half</sub>) | T<sub>half</sub> = 4 years
                            </div>
                            <p className="text-[10px] text-gray-500 mt-1">
                              Like CPA stretching a pulse before amplification, the halving schedule "stretches" the
                              total supply release over 256 years while front-loading early incentives.
                            </p>
                          </div>

                          <div className="bg-black/20 rounded-lg p-3">
                            <div className="text-xs text-amber-400 font-semibold mb-2">Phase-Locked Consensus Timing</div>
                            <p className="text-[11px] text-gray-400 mb-2">
                              The K-Parameter framework models validator agreement as phase-locked oscillators — the
                              same mathematics governing mode-locked lasers that produce attosecond pulses:
                            </p>
                            <div className="text-xs font-mono text-amber-300 bg-black/30 rounded p-2 text-center">
                              Psi(t) = Sum_n E_n * exp(i*n*omega_rep*t + i*phi_n)
                            </div>
                            <p className="text-[10px] text-gray-500 mt-1">
                              When validators "mode-lock" (achieve consensus), the superposition of their timing
                              signals produces ultrasharp finality — sub-3-second deterministic agreement from the
                              constructive interference of N independent oscillators.
                            </p>
                          </div>
                        </div>

                        <div className="bg-black/20 rounded-lg p-3">
                          <div className="text-xs text-gray-500 mb-2 text-center font-mono">
                            Timescale Hierarchy: Attosecond (10<sup>-18</sup>s) electron dynamics &rarr; Femtosecond (10<sup>-15</sup>s) molecular bonds &rarr;
                            Nanosecond (10<sup>-9</sup>s) CPU clock &rarr; Second (10<sup>0</sup>s) block time &rarr; Gigasecond (10<sup>9</sup>s) halving era
                          </div>
                          <p className="text-[10px] text-gray-500 text-center">
                            The emission controller operates across 27 orders of magnitude in timescale —
                            from nanosecond hash computations to gigasecond halving eras — unified by the same
                            time-energy reciprocity that governs ultrafast laser physics.
                          </p>
                        </div>
                      </div>

                      {/* Whitepaper Link */}
                      <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 text-center">
                        <a
                          href="https://quillon.xyz/downloads/qug-emission-economics-whitepaper.pdf"
                          target="_blank"
                          className="text-amber-400 hover:text-amber-300 underline font-semibold"
                        >
                          Read the Full Emission Economics Whitepaper (PDF, 16 pages)
                        </a>
                        <p className="text-xs text-gray-500 mt-1">Mathematical proofs, 256-year simulations, security analysis, and comparison with Bitcoin</p>
                      </div>
                    </>
                  )}

                  {activeTab === 'adoption' && (
                    <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                      <h3 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-green-400" />
                        Three-Layer Adoption Framework
                        <BigTooltip
                          title="Why Three Layers?"
                          explanation="Not all token usage is equal. Someone holding long-term is different from someone trading daily. This framework measures adoption across 3 use cases to get a complete picture of network health."
                        >
                          <span></span>
                        </BigTooltip>
                      </h3>

                      <div className="space-y-4">
                        {/* Layer 1 */}
                        <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/20">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <Wallet className="w-5 h-5 text-cyan-400" />
                              <BigTooltip
                                title="Layer 1: HODLers & Stakers"
                                explanation="People who buy and HOLD QUG long-term, often staking to earn rewards. These are your core believers who see QUG as a store of value - like a savings account. This layer has the highest weight (50%) because long-term holders provide stability."
                                example="Think: People who bought Bitcoin in 2015 and never sold"
                              >
                                <span className="font-medium text-white">Layer 1: Savings & Staking</span>
                              </BigTooltip>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300">50% weight</span>
                            </div>
                            <span className="text-2xl font-bold k-law-value" style={{ color: '#22d3ee' }}>{(data.three_layer_adoption.layer1_savings * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
                              style={{ width: `${data.three_layer_adoption.layer1_savings * 100}%` }}
                            />
                          </div>
                          <p className="mt-2 text-xs text-gray-400">Long-term holders staking for rewards</p>
                        </div>

                        {/* Layer 2 */}
                        <div className="p-4 rounded-xl bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <Activity className="w-5 h-5 text-purple-400" />
                              <BigTooltip
                                title="Layer 2: Active Users"
                                explanation="People actually USING QUG for payments and transfers. These users treat QUG like cash - sending money, paying for things, settling debts. This shows real-world utility beyond just speculation."
                                example="Think: Paying a friend, buying coffee, settling invoices"
                              >
                                <span className="font-medium text-white">Layer 2: Settlement & Payments</span>
                              </BigTooltip>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300">30% weight</span>
                            </div>
                            <span className="text-2xl font-bold k-law-value" style={{ color: '#c084fc' }}>{(data.three_layer_adoption.layer2_settlement * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                              style={{ width: `${data.three_layer_adoption.layer2_settlement * 100}%` }}
                            />
                          </div>
                          <p className="mt-2 text-xs text-gray-400">Transaction utility for payments</p>
                        </div>

                        {/* Layer 3 */}
                        <div className="p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <Anchor className="w-5 h-5 text-green-400" />
                              <BigTooltip
                                title="Layer 3: DeFi Power Users"
                                explanation="Advanced users putting QUG to work in DeFi. They provide liquidity to DEX pools, use QUG as loan collateral, or lock it in yield strategies. These users create the financial infrastructure that makes everything else work."
                                example="Think: Liquidity providers, borrowers/lenders, yield farmers"
                              >
                                <span className="font-medium text-white">Layer 3: Collateral & DeFi</span>
                              </BigTooltip>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-300">20% weight</span>
                            </div>
                            <span className="text-2xl font-bold k-law-value" style={{ color: '#4ade80' }}>{(data.three_layer_adoption.layer3_collateral * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-500"
                              style={{ width: `${data.three_layer_adoption.layer3_collateral * 100}%` }}
                            />
                          </div>
                          <p className="mt-2 text-xs text-gray-400">DeFi TVL, lending, liquidity provision</p>
                        </div>
                      </div>

                      {/* Composite */}
                      <div className="mt-6 p-5 rounded-xl bg-gradient-to-r from-cyan-900/40 to-blue-900/40 border border-cyan-500/40">
                        <div className="flex items-center justify-between">
                          <BigTooltip
                            title="Total Adoption Score"
                            explanation="This combines all three layers into one number. It's a weighted average: 50% from HODLers, 30% from active users, 20% from DeFi. This final score is what gets compared to the equilibrium ceiling to calculate network health."
                            example="5.2% adoption = The network has reached 5.2% of its potential user base"
                          >
                            <span className="text-lg font-semibold text-white">Composite Adoption (A_t)</span>
                          </BigTooltip>
                          <span className="text-4xl font-bold text-cyan-400">{(data.three_layer_adoption.composite_adoption * 100).toFixed(1)}%</span>
                        </div>
                        <p className="mt-3 text-sm text-gray-400 font-mono bg-black/20 p-2 rounded">
                          = 0.50×{(data.three_layer_adoption.layer1_savings).toFixed(2)} + 0.30×{(data.three_layer_adoption.layer2_settlement).toFixed(2)} + 0.20×{(data.three_layer_adoption.layer3_collateral).toFixed(2)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Holders Tab */}
                  {activeTab === 'holders' && (
                    <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                      <h3 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
                        <Users className="w-5 h-5 text-blue-400" />
                        Holder Distribution
                        <span className="ml-auto text-cyan-400 font-normal">{formatNumber(data.total_holders)} wallets</span>
                      </h3>

                      <div className="space-y-2">
                        {data.holder_distribution.map((cohort, i) => (
                          <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors">
                            <span className="text-2xl w-10 text-center">{cohort.emoji}</span>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-medium text-white">{cohort.name}</span>
                                <span className="text-sm text-gray-400">{cohort.range}</span>
                              </div>
                              <div className="flex items-center gap-4 text-xs">
                                <span className="text-cyan-400">{formatNumber(cohort.holder_count)} ({cohort.percentage_holders.toFixed(1)}%)</span>
                                <span className="text-purple-400">{formatNumber(cohort.total_balance)} QUG ({cohort.percentage_supply.toFixed(1)}%)</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Gini */}
                      <div className="mt-5 p-4 rounded-xl bg-slate-800/60 border border-orange-500/30">
                        <div className="flex items-center justify-between mb-3">
                          <BigTooltip
                            title="Wealth Distribution Score"
                            explanation="The Gini coefficient measures how evenly tokens are distributed. 0 = perfectly equal (everyone has exactly the same). 1 = one person owns everything. Most crypto projects are 0.6-0.9 (very unequal). Lower is generally better for decentralization."
                            example="0.3 = Fairly equal (like Sweden) | 0.9 = Very unequal (one whale holds most)"
                          >
                            <span className="text-sm text-gray-200">Gini Coefficient (Wealth Inequality)</span>
                          </BigTooltip>
                          <span className="font-bold text-2xl text-orange-300">{data.gini_coefficient.toFixed(3)}</span>
                        </div>
                        <div className="relative h-3 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-emerald-500 via-amber-400 to-red-500 rounded-full"
                            style={{ width: `${data.gini_coefficient * 100}%` }}
                          />
                        </div>
                        <div className="flex justify-between mt-2 text-xs text-gray-500">
                          <span>0 Equal</span>
                          <span>0.4 Typical</span>
                          <span>0.7 High</span>
                          <span>1 Monopoly</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* QCREDIT Yield Vault Tab */}
                  {activeTab === 'qcredit' && (
                    <>
                      {qcreditLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <RefreshCw className="w-8 h-8 text-amber-400 animate-spin" />
                        </div>
                      ) : qcreditStatus ? (
                        <div className="space-y-4">
                          {/* Header */}
                          <div className="text-center p-4 bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-xl">
                            <h3 className="text-lg font-bold bg-gradient-to-r from-amber-400 to-orange-300 bg-clip-text text-transparent">Quillon Credit (QCREDIT)</h3>
                            <p className="text-xs text-amber-300/60 mt-1">Lock QUG 1:1 to mint QCREDIT &mdash; earn tiered yield</p>
                          </div>

                          {/* TVL & Stats */}
                          <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-slate-800/50 border border-slate-700/30 rounded-xl">
                              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Total Value Locked</div>
                              <div className="text-lg font-bold text-white font-mono mt-1">{parseFloat(qcreditStatus.total_locked).toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG</div>
                            </div>
                            <div className="p-3 bg-slate-800/50 border border-slate-700/30 rounded-xl">
                              <div className="text-[10px] text-slate-500 uppercase tracking-wider">QCREDIT Supply</div>
                              <div className="text-lg font-bold text-amber-400 font-mono mt-1">{parseFloat(qcreditStatus.total_qcredit_supply).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                            </div>
                            <div className="p-3 bg-slate-800/50 border border-slate-700/30 rounded-xl">
                              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Protocol Reserve</div>
                              <div className="text-lg font-bold text-emerald-400 font-mono mt-1">{parseFloat(qcreditStatus.protocol_reserve).toLocaleString(undefined, { maximumFractionDigits: 2 })} QUG</div>
                            </div>
                            <div className="p-3 bg-slate-800/50 border border-slate-700/30 rounded-xl">
                              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Positions</div>
                              <div className="text-lg font-bold text-cyan-400 font-mono mt-1">{qcreditStatus.position_count}</div>
                            </div>
                          </div>

                          {/* Yield Tiers */}
                          <div>
                            <h4 className="text-sm font-bold text-slate-300 mb-2">Yield Tiers</h4>
                            <div className="grid grid-cols-2 gap-2">
                              {(qcreditStatus.tiers || []).map((tier: any, i: number) => {
                                const colors = [
                                  { bg: 'from-amber-700/20 to-amber-600/10', border: 'border-amber-600/30', text: 'text-amber-400' },
                                  { bg: 'from-slate-400/20 to-slate-300/10', border: 'border-slate-400/30', text: 'text-slate-300' },
                                  { bg: 'from-yellow-500/20 to-yellow-400/10', border: 'border-yellow-500/30', text: 'text-yellow-400' },
                                  { bg: 'from-cyan-400/20 to-blue-400/10', border: 'border-cyan-400/30', text: 'text-cyan-300' },
                                ][i] || { bg: 'from-slate-700/20 to-slate-600/10', border: 'border-slate-600/30', text: 'text-slate-400' };
                                return (
                                  <div key={tier.name} className={`p-3 bg-gradient-to-r ${colors.bg} border ${colors.border} rounded-xl`}>
                                    <div className="flex justify-between items-center">
                                      <span className={`font-bold ${colors.text}`}>{tier.name}</span>
                                      <span className="text-white font-bold text-lg">{tier.apy_percent}%</span>
                                    </div>
                                    <div className="text-[10px] text-slate-500 mt-1">{tier.lock_days} day lock</div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* How It Works */}
                          <div className="p-3 bg-slate-800/30 border border-slate-700/20 rounded-xl">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">How It Works</h4>
                            <div className="space-y-1.5 text-xs text-slate-500">
                              <div className="flex items-center gap-2"><span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">1</span> Lock QUG in a tier (7-180 days)</div>
                              <div className="flex items-center gap-2"><span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">2</span> Receive QCREDIT 1:1 (tradeable on DEX)</div>
                              <div className="flex items-center gap-2"><span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">3</span> Earn yield (claim anytime, paid in QUG)</div>
                              <div className="flex items-center gap-2"><span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold">4</span> Unlock after lock period (burn QCREDIT, get QUG back)</div>
                            </div>
                          </div>

                          <div className="p-2 bg-amber-500/5 border border-amber-500/10 rounded-lg text-center">
                            <span className="text-[10px] text-amber-300/60">Lock/unlock via the Wallet screen</span>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center py-12">
                          <div className="text-center">
                            <Zap className="w-12 h-12 text-amber-400/40 mx-auto mb-3" />
                            <p className="text-slate-500">QCREDIT vault loading...</p>
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* QUGUSD Tab */}
                  {activeTab === 'stablecoin' && (
                    <>
                      {stablecoinLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400" />
                        </div>
                      ) : stablecoinError || !stablecoinData ? (
                        <div className="flex flex-col items-center justify-center py-12 text-center">
                          <AlertCircle className="w-12 h-12 text-yellow-400 mb-4" />
                          <p className="text-gray-400 mb-2">{stablecoinError || 'Stablecoin data not available'}</p>
                          <p className="text-sm text-gray-500 mb-4">Server may need restart with latest code.</p>
                          <button
                            onClick={fetchStablecoinData}
                            className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-colors flex items-center gap-2"
                          >
                            <RefreshCw className="w-4 h-4" />
                            Retry
                          </button>
                        </div>
                      ) : (
                        <>
                          {/* Why $1 = $1 */}
                          <div className="p-5 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20">
                            <div className="flex items-center justify-between mb-4">
                              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                <Anchor className="w-5 h-5 text-green-400" />
                                Why 1 QUGUSD = $1 USD
                              </h3>
                              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                stablecoinData.system_health === 'Healthy' ? 'bg-green-500/20 text-green-400' :
                                stablecoinData.system_health === 'Warning' ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-gray-500/20 text-gray-400'
                              }`}>
                                {stablecoinData.system_health}
                              </span>
                            </div>
                            <p className="text-gray-300 mb-4 bg-black/20 p-3 rounded-lg">{stablecoinData.health_description}</p>

                            {/* Backing Bar */}
                            <div className="relative h-6 bg-white/10 rounded-full overflow-hidden mb-2">
                              <div
                                className="h-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-500"
                                style={{ width: `${Math.min(stablecoinData.backing_ratio * 100, 100)}%` }}
                              />
                              <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-white">
                                {(stablecoinData.backing_ratio * 100).toFixed(0)}% Backed
                              </div>
                            </div>
                            <div className="flex justify-between text-xs text-gray-500">
                              <span>0%</span>
                              <span className="text-green-400">100% Min</span>
                              <span className="text-cyan-400">150% Target</span>
                            </div>
                          </div>

                          {/* How Peg Works */}
                          <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                            <h3 className="text-lg font-semibold text-white mb-4">How the Peg Works</h3>
                            <p className="text-sm text-gray-400 mb-4">{stablecoinData.peg_mechanism.peg_mechanism}</p>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                              <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20 text-center">
                                <BigTooltip
                                  title="Why 150% Collateral?"
                                  explanation="To mint $100 QUGUSD, you must lock $150 worth of QUG. This extra 50% buffer protects against price drops. If QUG price falls, there's still enough collateral to back every QUGUSD. Think of it like a security deposit."
                                  example="Want 100 QUGUSD? Lock $150 of QUG as insurance"
                                >
                                  <p className="text-2xl font-bold text-green-400">{(stablecoinData.peg_mechanism.min_collateral_ratio * 100).toFixed(0)}%</p>
                                  <p className="text-xs text-gray-400 mt-1">Min Collateral</p>
                                </BigTooltip>
                              </div>
                              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
                                <BigTooltip
                                  title="Liquidation = Safety Net"
                                  explanation="If your collateral drops below 110% (due to QUG price falling), anyone can liquidate your position to protect the system. Your QUG gets sold to repay the QUGUSD and maintain the peg. Always keep collateral above 150% to be safe!"
                                  example="Collateral at 105%? You'll get liquidated. Stay above 150% to be safe."
                                >
                                  <p className="text-2xl font-bold text-red-400">{(stablecoinData.peg_mechanism.liquidation_ratio * 100).toFixed(0)}%</p>
                                  <p className="text-xs text-gray-400 mt-1">Liquidation</p>
                                </BigTooltip>
                              </div>
                              <div className="p-4 rounded-xl bg-amber-900/30 border border-amber-500/30 text-center">
                                <BigTooltip
                                  title="Liquidator Reward"
                                  explanation="People who help liquidate undercollateralized positions earn a 5% bonus. This incentivizes the community to keep the system healthy. Without this reward, no one would bother helping maintain the peg."
                                  example="Liquidate a $1000 position = Earn $50 bonus"
                                >
                                  <p className="text-2xl font-bold text-amber-300">{(stablecoinData.peg_mechanism.liquidation_bonus * 100).toFixed(0)}%</p>
                                  <p className="text-xs text-gray-300 mt-1">Liquidator Bonus</p>
                                </BigTooltip>
                              </div>
                              <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-center">
                                <BigTooltip
                                  title="Price Manipulation Protection"
                                  explanation="If the price oracle reports a change greater than 20% in a single update, the system rejects it. This prevents flash loan attacks or oracle manipulation from crashing the system."
                                  example="Oracle says QUG jumped 50%? Rejected. Must happen gradually."
                                >
                                  <p className="text-2xl font-bold text-blue-400">{stablecoinData.peg_mechanism.circuit_breaker_pct}%</p>
                                  <p className="text-xs text-gray-400 mt-1">Circuit Breaker</p>
                                </BigTooltip>
                              </div>
                            </div>
                          </div>

                          {/* Live Data */}
                          <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                              <Wallet className="w-5 h-5 text-purple-400" />
                              Live Blockchain Data
                            </h3>

                            <div className="grid grid-cols-2 gap-4 mb-4">
                              <div className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/20">
                                <p className="text-2xl font-bold k-law-value" style={{ color: '#ffffff' }}>${stablecoinData.backing.total_qugusd_supply.toLocaleString(undefined, {maximumFractionDigits: 2})}</p>
                                <p className="text-sm text-gray-400">QUGUSD in Circulation</p>
                              </div>
                              <div className="p-4 rounded-xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20">
                                <p className="text-2xl font-bold k-law-value" style={{ color: '#22d3ee' }}>${stablecoinData.backing.total_collateral_value_usd.toLocaleString(undefined, {maximumFractionDigits: 2})}</p>
                                <p className="text-sm text-gray-400">Total Collateral Value</p>
                              </div>
                            </div>

                            <div className="grid grid-cols-4 gap-3">
                              <div className="p-3 rounded-lg bg-white/5 text-center">
                                <BigTooltip
                                  title="Locked Collateral"
                                  explanation="Total QUG tokens that are locked as collateral backing QUGUSD. This QUG is held in smart contracts and can only be released when the corresponding QUGUSD is repaid."
                                  example="1M QUG locked = $1M+ worth of backing for QUGUSD"
                                >
                                  <p className="text-lg font-bold text-white">{stablecoinData.backing.total_qug_collateral.toLocaleString(undefined, {maximumFractionDigits: 0})}</p>
                                  <p className="text-xs text-gray-400">QUG Locked</p>
                                </BigTooltip>
                              </div>
                              <div className="p-3 rounded-lg bg-white/5 text-center">
                                <BigTooltip
                                  title="Oracle Price"
                                  explanation="The current market price of QUG according to the price oracle. This determines how much QUGUSD you can mint and whether positions are at risk of liquidation."
                                  example="QUG = $1.50 means your 100 QUG = $150 collateral value"
                                >
                                  <p className="text-lg font-bold text-cyan-400">${stablecoinData.backing.qug_price_usd.toFixed(2)}</p>
                                  <p className="text-xs text-gray-400">QUG Price</p>
                                </BigTooltip>
                              </div>
                              <div className="p-3 rounded-lg bg-white/5 text-center">
                                <BigTooltip
                                  title="Overall System Health"
                                  explanation="The average collateral ratio across ALL positions. Should be well above 150%. Higher = safer for the whole system. If this drops too low, more positions are at risk of liquidation."
                                  example="200% system ratio = Very healthy buffer"
                                >
                                  <p className="text-lg font-bold text-green-400">{(stablecoinData.backing.system_collateral_ratio * 100).toFixed(0)}%</p>
                                  <p className="text-xs text-gray-400">System Ratio</p>
                                </BigTooltip>
                              </div>
                              <div className="p-3 rounded-lg bg-white/5 text-center">
                                <BigTooltip
                                  title="Active Positions"
                                  explanation="Number of users who have locked QUG to mint QUGUSD. Each position represents someone who deposited collateral. More positions = more decentralized backing."
                                  example="50 CDPs = 50 different users backing the stablecoin"
                                >
                                  <p className="text-lg font-bold text-purple-400">{stablecoinData.backing.active_positions}</p>
                                  <p className="text-xs text-gray-400">Active CDPs</p>
                                </BigTooltip>
                              </div>
                            </div>
                          </div>
                        </>
                      )}
                    </>
                  )}

                  {/* Checkpoints Tab */}
                  {activeTab === 'checkpoints' && (
                    <>
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
                          <Target className="w-5 h-5 text-green-400" />
                          Adoption Checkpoints (Falsifiable Predictions)
                        </h3>

                        <div className="space-y-3">
                          {data.checkpoints.map((cp, i) => (
                            <div key={i} className="flex items-center gap-4 p-4 rounded-xl bg-white/5 border border-white/10">
                              <div className="text-center min-w-[60px]">
                                <span className="text-2xl font-bold k-law-value" style={{ color: '#ffffff' }}>{cp.target_year}</span>
                              </div>
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <div className="h-3 flex-1 bg-white/10 rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full transition-all duration-500"
                                      style={{ width: `${cp.predicted_adoption * 100}%` }}
                                    />
                                  </div>
                                  <span className="text-sm font-bold text-cyan-400 min-w-[50px] text-right">
                                    {(cp.predicted_adoption * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <p className="text-sm text-gray-400">Target: {formatNumber(cp.predicted_holders)} holders</p>
                              </div>
                              <div className="text-3xl">
                                {cp.status === 'Future' && '⏳'}
                                {cp.status === 'Active' && '🔵'}
                                {cp.status === 'Met' && '✅'}
                                {cp.status === 'Missed' && '❌'}
                                {cp.status === 'Exceeded' && '🚀'}
                              </div>
                            </div>
                          ))}
                        </div>

                        <div className="mt-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-start gap-2">
                          <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                          <p className="text-sm text-blue-300">
                            These predictions are <strong>falsifiable</strong>. As each date passes, actual metrics will be compared against predictions to validate the K-Law model.
                          </p>
                        </div>
                      </div>

                      {/* Supply Stats */}
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                          <BigTooltip
                            title="Maximum Supply Cap"
                            explanation="The absolute maximum number of QUG tokens that will EVER exist. This is hard-coded into the protocol and cannot be changed. Unlike traditional currencies that can print more money, this cap ensures scarcity."
                            example="21M for Bitcoin, 100M for QUG - once reached, no more can be created"
                          >
                            <p className="text-2xl font-bold k-law-value" style={{ color: '#ffffff' }}>{formatNumber(data.total_supply)}</p>
                            <p className="text-xs text-gray-400">Max Supply</p>
                          </BigTooltip>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                          <BigTooltip
                            title="Available Right Now"
                            explanation="Tokens that are actually tradeable in the market right now. Excludes tokens that are locked in vesting schedules, held by team, or reserved for future use. This is what affects daily trading."
                            example="50M circulating out of 100M max = 50% of supply available"
                          >
                            <p className="text-2xl font-bold k-law-value" style={{ color: '#22d3ee' }}>{formatNumber(data.circulating_supply)}</p>
                            <p className="text-xs text-gray-400">Circulating</p>
                          </BigTooltip>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
                          <BigTooltip
                            title="Tokens Earning Rewards"
                            explanation="Percentage of tokens that holders have 'staked' to earn rewards. Staked tokens are locked and can't be sold easily. High staking = strong holder confidence and reduced selling pressure."
                            example="40% staked = 40% of tokens locked earning rewards, not for sale"
                          >
                            <p className="text-2xl font-bold k-law-value" style={{ color: '#c084fc' }}>{data.staking_percentage.toFixed(1)}%</p>
                            <p className="text-xs text-gray-400">Staked</p>
                          </BigTooltip>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Graphs Tab - v3.4.15: Beautiful Interactive Charts */}
                  {activeTab === 'graphs' && (
                    <>
                      {/* K-Law Adoption Curve */}
                      <div className="p-5 rounded-xl bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-purple-500/20">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <LineChart className="w-5 h-5 text-purple-400" />
                          K-Law Adoption Curve (S-Curve)
                        </h3>
                        <div className="relative h-64 bg-black/30 rounded-xl p-4 overflow-hidden">
                          <svg viewBox="0 0 400 200" className="w-full h-full">
                            {/* Grid lines */}
                            {[0, 25, 50, 75, 100].map((y) => (
                              <g key={y}>
                                <line x1="40" y1={180 - y * 1.6} x2="380" y2={180 - y * 1.6} stroke="rgba(255,255,255,0.1)" strokeDasharray="4" />
                                <text x="35" y={184 - y * 1.6} fill="#6b7280" fontSize="8" textAnchor="end">{y}%</text>
                              </g>
                            ))}
                            {/* X-axis labels - proportionally spaced (2025-2035 = 10 years) */}
                            {[2025, 2027, 2029, 2031, 2033, 2035].map((year) => {
                              const xPos = 60 + ((year - 2025) / 10) * 320; // 320px span for 10 years
                              return (
                                <text key={year} x={xPos} y="195" fill="#6b7280" fontSize="8" textAnchor="middle">{year}</text>
                              );
                            })}

                            {/* Animated S-curve path */}
                            <motion.path
                              d="M 50,175 C 100,175 120,172 150,165 C 180,155 200,140 230,110 C 260,80 290,55 320,35 C 350,20 370,15 380,12"
                              fill="none"
                              stroke="url(#adoptionGradient)"
                              strokeWidth="3"
                              strokeLinecap="round"
                              initial={{ pathLength: 0 }}
                              animate={{ pathLength: 1 }}
                              transition={{ duration: 2, ease: "easeOut" }}
                            />

                            {/* Current position dot - X based on year (2026), Y based on adoption rate */}
                            {(() => {
                              const currentYear = new Date().getFullYear(); // 2026
                              const startYear = 2025;
                              const endYear = 2035;
                              const xStart = 60;
                              const xEnd = 380;
                              // Calculate X position based on current year
                              const yearProgress = (currentYear - startYear) / (endYear - startYear);
                              const cx = xStart + yearProgress * (xEnd - xStart);
                              // Y position based on adoption rate
                              const cy = 180 - (data.kristensen_ratio.current_adoption * 100) * 1.6;
                              return (
                                <motion.circle
                                  cx={cx}
                                  cy={cy}
                                  r="6"
                                  fill="#22d3ee"
                                  initial={{ scale: 0 }}
                                  animate={{ scale: [1, 1.3, 1] }}
                                  transition={{ duration: 1.5, repeat: Infinity }}
                                />
                              );
                            })()}

                            {/* Equilibrium line */}
                            <motion.line
                              x1="40"
                              y1={180 - (data.kristensen_ratio.equilibrium_ceiling * 100) * 1.6}
                              x2="380"
                              y2={180 - (data.kristensen_ratio.equilibrium_ceiling * 100) * 1.6}
                              stroke="#c084fc"
                              strokeWidth="2"
                              strokeDasharray="8 4"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 1 }}
                            />

                            {/* Gradient definitions */}
                            <defs>
                              <linearGradient id="adoptionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#22d3ee" />
                                <stop offset="50%" stopColor="#c084fc" />
                                <stop offset="100%" stopColor="#22c55e" />
                              </linearGradient>
                            </defs>
                          </svg>

                          {/* Legend */}
                          <div className="absolute bottom-2 right-4 flex gap-4 text-xs">
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 rounded-full bg-cyan-400"></div>
                              <span className="text-gray-400">Current</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-0.5 bg-purple-400"></div>
                              <span className="text-gray-400">Equilibrium</span>
                            </div>
                          </div>
                        </div>
                        <p className="text-xs text-gray-500 mt-2 text-center">Logistic adoption curve: A*(t) = K / (1 + μ·e^(-λ·Ω))</p>
                      </div>

                      {/* Flow Density Radar Chart */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Waves className="w-5 h-5 text-cyan-400" />
                          Network Flow Radar
                        </h3>
                        <div className="relative h-64 bg-black/30 rounded-xl flex items-center justify-center">
                          <svg viewBox="0 0 300 300" className="w-64 h-64">
                            {/* Radar background circles */}
                            {[1, 0.75, 0.5, 0.25].map((scale) => (
                              <circle
                                key={scale}
                                cx="150"
                                cy="150"
                                r={100 * scale}
                                fill="none"
                                stroke="rgba(255,255,255,0.1)"
                                strokeWidth="1"
                              />
                            ))}

                            {/* Radar axes */}
                            {['Staking', 'DeFi', 'Treasury', 'Unlocks', 'Exchange'].map((label, i) => {
                              const angle = (Math.PI * 2 * i) / 5 - Math.PI / 2;
                              const x = 150 + Math.cos(angle) * 120;
                              const y = 150 + Math.sin(angle) * 120;
                              const lineX = 150 + Math.cos(angle) * 100;
                              const lineY = 150 + Math.sin(angle) * 100;
                              return (
                                <g key={label}>
                                  <line x1="150" y1="150" x2={lineX} y2={lineY} stroke="rgba(255,255,255,0.2)" />
                                  <text x={x} y={y} fill="#9ca3af" fontSize="10" textAnchor="middle" dominantBaseline="middle">
                                    {label}
                                  </text>
                                </g>
                              );
                            })}

                            {/* Data polygon */}
                            <motion.polygon
                              points={(() => {
                                const flows = [
                                  data.current_flow.staking_flow,
                                  data.current_flow.defi_flow,
                                  data.current_flow.treasury_flow,
                                  data.current_flow.unlock_flow,
                                  data.current_flow.exchange_flow,
                                ];
                                return flows.map((flow, i) => {
                                  const angle = (Math.PI * 2 * i) / 5 - Math.PI / 2;
                                  const r = Math.min(flow * 200, 100); // Scale to max 100
                                  const x = 150 + Math.cos(angle) * r;
                                  const y = 150 + Math.sin(angle) * r;
                                  return `${x},${y}`;
                                }).join(' ');
                              })()}
                              fill="rgba(34, 211, 238, 0.3)"
                              stroke="#22d3ee"
                              strokeWidth="2"
                              initial={{ opacity: 0, scale: 0.5 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ duration: 0.8, ease: "easeOut" }}
                            />

                            {/* Data points */}
                            {[
                              data.current_flow.staking_flow,
                              data.current_flow.defi_flow,
                              data.current_flow.treasury_flow,
                              data.current_flow.unlock_flow,
                              data.current_flow.exchange_flow,
                            ].map((flow, i) => {
                              const angle = (Math.PI * 2 * i) / 5 - Math.PI / 2;
                              const r = Math.min(flow * 200, 100);
                              const x = 150 + Math.cos(angle) * r;
                              const y = 150 + Math.sin(angle) * r;
                              return (
                                <motion.circle
                                  key={i}
                                  cx={x}
                                  cy={y}
                                  r="5"
                                  fill="#22d3ee"
                                  initial={{ scale: 0 }}
                                  animate={{ scale: 1 }}
                                  transition={{ delay: 0.8 + i * 0.1 }}
                                />
                              );
                            })}
                          </svg>

                          {/* Composite Omega display */}
                          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-center">
                            <div className="text-2xl font-bold text-cyan-400">Ω = {data.current_flow.composite_omega.toFixed(4)}</div>
                            <div className="text-xs text-gray-500">Composite Flow Density</div>
                          </div>
                        </div>
                      </div>

                      {/* Holder Distribution Pie Chart */}
                      <div className="p-5 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Users className="w-5 h-5 text-green-400" />
                          Holder Distribution
                        </h3>
                        <div className="flex items-center gap-8">
                          {/* Pie chart */}
                          <div className="relative w-48 h-48">
                            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                              {(() => {
                                const colors = ['#22d3ee', '#c084fc', '#22c55e', '#eab308', '#ec4899', '#f97316'];
                                let cumulativePercent = 0;
                                return data.holder_distribution.slice(0, 6).map((cohort, i) => {
                                  const percent = cohort.percentage_supply / 100;
                                  const startAngle = cumulativePercent * 360;
                                  cumulativePercent += percent;
                                  const endAngle = cumulativePercent * 360;

                                  // SVG arc path
                                  const largeArcFlag = percent > 0.5 ? 1 : 0;
                                  const startX = 50 + 40 * Math.cos((startAngle - 90) * Math.PI / 180);
                                  const startY = 50 + 40 * Math.sin((startAngle - 90) * Math.PI / 180);
                                  const endX = 50 + 40 * Math.cos((endAngle - 90) * Math.PI / 180);
                                  const endY = 50 + 40 * Math.sin((endAngle - 90) * Math.PI / 180);

                                  return (
                                    <motion.path
                                      key={i}
                                      d={`M 50 50 L ${startX} ${startY} A 40 40 0 ${largeArcFlag} 1 ${endX} ${endY} Z`}
                                      fill={colors[i]}
                                      initial={{ opacity: 0 }}
                                      animate={{ opacity: 0.9 }}
                                      transition={{ delay: i * 0.1 }}
                                      className="hover:opacity-100 cursor-pointer transition-opacity"
                                    />
                                  );
                                });
                              })()}
                            </svg>
                            <div className="absolute inset-0 flex items-center justify-center">
                              <div className="text-center">
                                <div className="text-xl font-bold text-white">{formatNumber(data.total_holders)}</div>
                                <div className="text-xs text-gray-400">Total Holders</div>
                              </div>
                            </div>
                          </div>

                          {/* Legend */}
                          <div className="flex-1 space-y-2">
                            {data.holder_distribution.slice(0, 6).map((cohort, i) => {
                              const colors = ['bg-cyan-400', 'bg-purple-400', 'bg-green-400', 'bg-yellow-400', 'bg-pink-400', 'bg-orange-400'];
                              return (
                                <div key={i} className="flex items-center gap-2 text-sm">
                                  <div className={`w-3 h-3 rounded-full ${colors[i]}`}></div>
                                  <span className="text-gray-300 flex-1">{cohort.emoji} {cohort.name}</span>
                                  <span className="text-gray-400">{cohort.percentage_supply.toFixed(1)}%</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      {/* Gini Index Bar */}
                      <div className="p-5 rounded-xl bg-gradient-to-r from-orange-500/10 to-red-500/10 border border-orange-500/20">
                        <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                          <BarChart3 className="w-5 h-5 text-orange-400" />
                          Wealth Distribution Index
                        </h3>
                        <div className="relative h-12 bg-black/30 rounded-xl overflow-hidden">
                          {/* Gradient bar */}
                          <div className="absolute inset-0 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 opacity-30" />

                          {/* Animated indicator */}
                          <motion.div
                            className="absolute top-0 h-full w-1 bg-white shadow-lg shadow-white/50"
                            initial={{ left: '0%' }}
                            animate={{ left: `${data.gini_coefficient * 100}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                          />

                          {/* Current value */}
                          <motion.div
                            className="absolute -top-1 transform -translate-x-1/2"
                            initial={{ left: '0%' }}
                            animate={{ left: `${data.gini_coefficient * 100}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                          >
                            <div className="bg-black/80 px-2 py-1 rounded text-sm font-bold text-orange-400 border border-orange-500/30">
                              {data.gini_coefficient.toFixed(3)}
                            </div>
                          </motion.div>

                          {/* Scale markers */}
                          <div className="absolute bottom-1 left-2 text-xs text-gray-400">0 Equal</div>
                          <div className="absolute bottom-1 right-2 text-xs text-gray-400">1 Monopoly</div>
                        </div>
                        <div className="flex justify-between mt-4 text-xs">
                          <div className="text-green-400">🌍 Sweden: 0.25</div>
                          <div className="text-yellow-400">🇺🇸 USA: 0.39</div>
                          <div className="text-orange-400">₿ Bitcoin: 0.88</div>
                          <div className="text-red-400">⚠️ Danger: 0.95+</div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ) : null}
            </div>

            {/* Footer */}
            <div className="flex-shrink-0 p-4 border-t border-white/10 flex items-center justify-between">
              <span className="text-xs text-gray-500">Powered by Water Robot Financial Intelligence</span>
              <span className="text-xs text-cyan-400 font-mono">A*_t = K / (1 + μ·e^(-λ·Ω_t))</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default FinanceModal;
