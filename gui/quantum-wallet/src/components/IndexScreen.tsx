import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PieChart,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Plus,
  Minus,
  BarChart3,
  Coins,
  Clock,
  Shield,
  ChevronRight,
  Info,
  ArrowUpRight,
  ArrowDownRight,
  Wallet,
  Settings2,
  Vote,
  History
} from 'lucide-react';

interface IndexComponent {
  symbol: string;
  name: string;
  weight: number;
  targetWeight: number;
  price: number;
  priceChange24h: number;
  holdings: number;
  value: number;
}

interface IndexFund {
  id: string;
  name: string;
  symbol: string;
  navPerShare: number;
  navChange24h: number;
  totalSupply: number;
  tvl: number;
  components: IndexComponent[];
  managementFee: number;
  methodology: 'market-cap' | 'equal-weight' | 'custom' | 'risk-adjusted';
  lastRebalance: number;
  nextRebalance: number;
  myShares: number;
  myValue: number;
}

// Demo data for index funds
const DEMO_INDEX_FUNDS: IndexFund[] = [
  {
    id: 'qnk10',
    name: 'QNK Top 10 — Quillon Native Edition',
    symbol: 'QNK10',
    navPerShare: 8682.0, // 3× QUG @ ~$2894 (matches backend handlers.rs:11578 NAV multiplier)
    navChange24h: 0.65,
    totalSupply: 1250000,
    tvl: 4341000,
    // v10.10.12: All 10 components are REAL Quillon-native tokens (was placeholder
    // QBTC/QETH/QSOL/QLINK/QDOT/QAVAX/QMATIC/QATOM/QUNI — none of those exist on
    // the chain). The UI was rendering only 2 of those because lookups failed.
    // Selection rationale: base asset (QUG), stablecoin floor (QUGUSD), 4 real
    // bridge wraps (wBTC/wETH/wZEC/wIRON), 2 native L2/L3 yield primitives
    // (QCREDIT/QSHARE), 2 commemoratives (CLAI/VAULT — real deployed tokens).
    components: [
      { symbol: 'QUG',     name: 'Quillon',            weight: 30, targetWeight: 30, price: 2894.00, priceChange24h:  0.5, holdings: 450,       value: 1302300 },
      { symbol: 'QUGUSD',  name: 'Quillon USD',        weight: 20, targetWeight: 20, price: 1.00,    priceChange24h:  0.0, holdings: 868200,    value: 868200 },
      { symbol: 'wBTC',    name: 'Wrapped Bitcoin',    weight: 12, targetWeight: 12, price: 100000,  priceChange24h: -0.3, holdings: 5.21,      value: 520920 },
      { symbol: 'wETH',    name: 'Wrapped Ethereum',   weight: 10, targetWeight: 10, price: 3500,    priceChange24h:  1.2, holdings: 124.03,    value: 434100 },
      { symbol: 'QSHARE',  name: 'Quillon Share',      weight:  8, targetWeight:  8, price: 100.00,  priceChange24h:  1.2, holdings: 3472.8,    value: 347280 },
      { symbol: 'QCREDIT', name: 'Quillon Credit',     weight:  7, targetWeight:  7, price: 1.12,    priceChange24h:  0.4, holdings: 271312,    value: 303870 },
      { symbol: 'wZEC',    name: 'Wrapped Zcash',      weight:  5, targetWeight:  5, price: 35,      priceChange24h:  2.1, holdings: 6201.5,    value: 217050 },
      { symbol: 'wIRON',   name: 'Wrapped Iron Fish',  weight:  4, targetWeight:  4, price: 12,      priceChange24h: -0.8, holdings: 14470,     value: 173640 },
      { symbol: 'CLAI',    name: 'Claude AI Commemorative', weight: 2, targetWeight: 2, price: 1.50,  priceChange24h:  4.5, holdings: 57880,     value: 86820 },
      { symbol: 'VAULT',   name: 'Quillon Vault',      weight:  2, targetWeight:  2, price: 0.85,    priceChange24h:  0.0, holdings: 102141,    value: 86820 },
    ],
    managementFee: 0.10, // 0.1% mint/redeem fee (matches handlers.rs:11586 fee_rate=0.999)
    methodology: 'market-cap',
    lastRebalance: Date.now() - 7 * 24 * 60 * 60 * 1000,
    nextRebalance: Date.now() + 7 * 24 * 60 * 60 * 1000,
    myShares: 125.5,
    myValue: 156.32,
  },
  {
    id: 'defi5',
    name: 'Stable Yield Index — Quillon Edition',
    symbol: 'DEFI5',
    navPerShare: 5788.0, // 2× QUG @ ~$2894 (matches backend handlers.rs:11578 NAV multiplier)
    navChange24h: 0.85,
    totalSupply: 500000,
    tvl: 2894000,
    // v10.10.13: real Quillon-native yield basket (was placeholder QAAVE/QCOMP/etc).
    // Backing tokens that actually exist on-chain and accrue real yield:
    //   QUGUSD — stablecoin floor, 1:1 USD peg via QUG-collateralized CDP
    //   QCREDIT — L2 yield vault (5-25% APY across Bronze→Platinum tiers)
    //   QSHARE — L3 treasury share with autonomous premium-arbitrage minting
    //   QUG — base asset, mining-secured proof-of-work emission
    //   wBTC — Bitcoin-bridge wrapped, external real-world floor exposure
    // Backend NAV currently uses 2× QUG multiplier (handlers.rs:11578) — moving
    // to a real basket-weighted NAV is a follow-up (TODO v10.11.x: index_nav.rs
    // computes weight-sum of reserves from the 5 underlying pools).
    components: [
      { symbol: 'QUGUSD', name: 'Quillon USD',     weight: 30, targetWeight: 30, price: 1.00,    priceChange24h:  0.0, holdings: 868200,    value: 868200 },
      { symbol: 'QCREDIT', name: 'Quillon Credit', weight: 25, targetWeight: 25, price: 1.12,    priceChange24h:  0.4, holdings: 645982.14, value: 723500 },
      { symbol: 'QSHARE',  name: 'Quillon Share',  weight: 20, targetWeight: 20, price: 100.00,  priceChange24h:  1.2, holdings: 5788,      value: 578800 },
      { symbol: 'QUG',     name: 'Quillon',        weight: 15, targetWeight: 15, price: 2894.00, priceChange24h:  0.5, holdings: 150,       value: 434100 },
      { symbol: 'wBTC',    name: 'Wrapped Bitcoin',weight: 10, targetWeight: 10, price: 100000,  priceChange24h: -0.3, holdings: 2.894,     value: 289400 },
    ],
    managementFee: 0.10, // 0.1% mint/redeem fee (matches handlers.rs:11586 fee_rate=0.999)
    methodology: 'risk-adjusted',
    lastRebalance: Date.now() - 14 * 24 * 60 * 60 * 1000,
    nextRebalance: Date.now() + 0.5 * 24 * 60 * 60 * 1000,
    myShares: 0,
    myValue: 0,
  },
];

export default function IndexScreen() {
  const [selectedFund, setSelectedFund] = useState<IndexFund | null>(DEMO_INDEX_FUNDS[0]);
  const [activeTab, setActiveTab] = useState<'overview' | 'components' | 'mint-redeem' | 'governance'>('overview');
  const [mintAmount, setMintAmount] = useState('');
  const [redeemAmount, setRedeemAmount] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);

  const formatCurrency = (value: number, decimals = 2) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
      return `$${(value / 1000).toFixed(2)}K`;
    }
    return `$${value.toFixed(decimals)}`;
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const getMethodologyLabel = (methodology: string) => {
    switch (methodology) {
      case 'market-cap': return 'Market Cap Weighted';
      case 'equal-weight': return 'Equal Weight';
      case 'custom': return 'Custom Weights';
      case 'risk-adjusted': return 'Risk Adjusted';
      default: return methodology;
    }
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 via-yellow-500 to-amber-600 bg-clip-text text-transparent">
            Index Funds
          </h1>
          <p className="text-amber-200/60 mt-1">Diversified exposure to top tokens</p>
        </div>
        <motion.button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl font-semibold"
          style={{
            background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(255, 215, 0, 0.15) 100%)',
            border: '2px solid rgba(212, 175, 55, 0.4)',
          }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Plus className="w-5 h-5 text-amber-400" />
          <span className="text-amber-100">Create Index</span>
        </motion.button>
      </div>

      {/* Portfolio Summary Card */}
      <motion.div
        className="p-6 rounded-2xl"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
          border: '2px solid rgba(212, 175, 55, 0.3)',
          boxShadow: '0 0 30px rgba(212, 175, 55, 0.1)',
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-lg font-semibold text-amber-100 mb-4 flex items-center gap-2">
          <Wallet className="w-5 h-5 text-amber-400" />
          My Index Portfolio
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-amber-200/60 text-sm">Total Value</p>
            <p className="text-2xl font-bold text-amber-100">
              {formatCurrency(DEMO_INDEX_FUNDS.reduce((sum, f) => sum + f.myValue, 0))}
            </p>
          </div>
          <div>
            <p className="text-amber-200/60 text-sm">Active Positions</p>
            <p className="text-2xl font-bold text-amber-100">
              {DEMO_INDEX_FUNDS.filter(f => f.myShares > 0).length}
            </p>
          </div>
          <div>
            <p className="text-amber-200/60 text-sm">24h Change</p>
            <p className="text-2xl font-bold text-green-400 flex items-center gap-1">
              <TrendingUp className="w-5 h-5" />
              +$5.39
            </p>
          </div>
        </div>
      </motion.div>

      {/* Index Funds List */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Fund Cards */}
        <div className="lg:col-span-1 space-y-4">
          <h3 className="text-lg font-semibold text-amber-100">Available Indices</h3>
          {DEMO_INDEX_FUNDS.map((fund) => (
            <motion.div
              key={fund.id}
              onClick={() => setSelectedFund(fund)}
              className={`p-4 rounded-xl cursor-pointer transition-all ${
                selectedFund?.id === fund.id ? 'ring-2 ring-amber-400' : ''
              }`}
              style={{
                background: selectedFund?.id === fund.id
                  ? 'linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(255, 215, 0, 0.1) 100%)'
                  : 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%)',
                border: '1px solid rgba(212, 175, 55, 0.2)',
              }}
              whileHover={{ scale: 1.02, x: 5 }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 100%)',
                    }}
                  >
                    <PieChart className="w-5 h-5 text-slate-900" />
                  </div>
                  <div>
                    <p className="font-semibold text-amber-100">{fund.symbol}</p>
                    <p className="text-xs text-amber-200/60">{fund.name}</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-amber-400/50" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-amber-200/60">NAV</p>
                  <p className="font-semibold text-amber-100">{formatCurrency(fund.navPerShare, 4)}</p>
                </div>
                <div>
                  <p className="text-xs text-amber-200/60">24h</p>
                  <p className={`font-semibold flex items-center gap-1 ${
                    fund.navChange24h >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {fund.navChange24h >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {formatPercent(fund.navChange24h)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-amber-200/60">TVL</p>
                  <p className="font-semibold text-amber-100">{formatCurrency(fund.tvl)}</p>
                </div>
                <div>
                  <p className="text-xs text-amber-200/60">My Holdings</p>
                  <p className="font-semibold text-amber-100">
                    {fund.myShares > 0 ? `${fund.myShares.toFixed(2)} shares` : '-'}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Selected Fund Details */}
        {selectedFund && (
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            key={selectedFund.id}
          >
            <div
              className="p-6 rounded-2xl h-full"
              style={{
                background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%)',
                border: '2px solid rgba(212, 175, 55, 0.3)',
              }}
            >
              {/* Fund Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div
                    className="w-14 h-14 rounded-xl flex items-center justify-center"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37 0%, #FFD700 100%)',
                      boxShadow: '0 0 20px rgba(212, 175, 55, 0.4)',
                    }}
                  >
                    <PieChart className="w-8 h-8 text-slate-900" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-amber-100">{selectedFund.name}</h2>
                    <p className="text-amber-200/60">{selectedFund.symbol} • {getMethodologyLabel(selectedFund.methodology)}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold text-amber-100">{formatCurrency(selectedFund.navPerShare, 4)}</p>
                  <p className={`flex items-center justify-end gap-1 ${
                    selectedFund.navChange24h >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {selectedFund.navChange24h >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    {formatPercent(selectedFund.navChange24h)} (24h)
                  </p>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-2 mb-6 border-b border-amber-500/20 pb-4">
                {([
                  { id: 'overview', label: 'Overview', icon: BarChart3 },
                  { id: 'components', label: 'Components', icon: PieChart },
                  { id: 'mint-redeem', label: 'Mint/Redeem', icon: Coins },
                  { id: 'governance', label: 'Governance', icon: Vote },
                ] as const).map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                        activeTab === tab.id
                          ? 'bg-amber-500/20 text-amber-100'
                          : 'text-amber-200/60 hover:text-amber-100'
                      }`}
                    >
                      <tab.icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  ))}
              </div>

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                {activeTab === 'overview' && (
                  <motion.div
                    key="overview"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-6"
                  >
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-xl bg-slate-800/50">
                        <p className="text-xs text-amber-200/60 flex items-center gap-1">
                          <Coins className="w-3 h-3" />
                          Total Supply
                        </p>
                        <p className="text-lg font-semibold text-amber-100 mt-1">
                          {selectedFund.totalSupply.toLocaleString()}
                        </p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-800/50">
                        <p className="text-xs text-amber-200/60 flex items-center gap-1">
                          <BarChart3 className="w-3 h-3" />
                          TVL
                        </p>
                        <p className="text-lg font-semibold text-amber-100 mt-1">
                          {formatCurrency(selectedFund.tvl)}
                        </p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-800/50">
                        <p className="text-xs text-amber-200/60 flex items-center gap-1">
                          <Shield className="w-3 h-3" />
                          Management Fee
                        </p>
                        <p className="text-lg font-semibold text-amber-100 mt-1">
                          {selectedFund.managementFee}% / year
                        </p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-800/50">
                        <p className="text-xs text-amber-200/60 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Next Rebalance
                        </p>
                        <p className="text-lg font-semibold text-amber-100 mt-1">
                          {Math.ceil((selectedFund.nextRebalance - Date.now()) / (24 * 60 * 60 * 1000))} days
                        </p>
                      </div>
                    </div>

                    {/* Weight Distribution Chart (Simplified) */}
                    <div>
                      <h3 className="text-lg font-semibold text-amber-100 mb-4">Weight Distribution</h3>
                      <div className="space-y-2">
                        {selectedFund.components.slice(0, 5).map((component) => (
                          <div key={component.symbol} className="flex items-center gap-3">
                            <span className="w-16 text-sm font-medium text-amber-200">{component.symbol}</span>
                            <div className="flex-1 h-4 bg-slate-700/50 rounded-full overflow-hidden">
                              <motion.div
                                className="h-full rounded-full"
                                style={{
                                  background: 'linear-gradient(90deg, #D4AF37, #FFD700)',
                                }}
                                initial={{ width: 0 }}
                                animate={{ width: `${component.weight}%` }}
                                transition={{ duration: 0.5, delay: 0.1 }}
                              />
                            </div>
                            <span className="w-14 text-sm text-amber-100 text-right">{component.weight.toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {activeTab === 'components' && (
                  <motion.div
                    key="components"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="text-left text-xs text-amber-200/60 border-b border-amber-500/20">
                            <th className="pb-3 font-medium">Token</th>
                            <th className="pb-3 font-medium">Weight</th>
                            <th className="pb-3 font-medium">Target</th>
                            <th className="pb-3 font-medium text-right">Price</th>
                            <th className="pb-3 font-medium text-right">24h</th>
                            <th className="pb-3 font-medium text-right">Holdings</th>
                            <th className="pb-3 font-medium text-right">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedFund.components.map((component, i) => (
                            <motion.tr
                              key={component.symbol}
                              className="border-b border-amber-500/10 hover:bg-amber-500/5"
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.05 }}
                            >
                              <td className="py-3">
                                <div className="flex items-center gap-2">
                                  <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center text-xs font-bold text-amber-400">
                                    {component.symbol.slice(0, 2)}
                                  </div>
                                  <div>
                                    <p className="font-medium text-amber-100">{component.symbol}</p>
                                    <p className="text-xs text-amber-200/60">{component.name}</p>
                                  </div>
                                </div>
                              </td>
                              <td className="py-3 text-amber-100">{component.weight.toFixed(1)}%</td>
                              <td className="py-3 text-amber-200/60">{component.targetWeight}%</td>
                              <td className="py-3 text-right text-amber-100">{formatCurrency(component.price)}</td>
                              <td className={`py-3 text-right ${component.priceChange24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {formatPercent(component.priceChange24h)}
                              </td>
                              <td className="py-3 text-right text-amber-100">{component.holdings.toLocaleString()}</td>
                              <td className="py-3 text-right text-amber-100">{formatCurrency(component.value)}</td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}

                {activeTab === 'mint-redeem' && (
                  <motion.div
                    key="mint-redeem"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-6"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Mint Section */}
                      <div className="p-5 rounded-xl bg-green-500/10 border border-green-500/30">
                        <h3 className="text-lg font-semibold text-green-400 flex items-center gap-2 mb-4">
                          <Plus className="w-5 h-5" />
                          Mint Shares
                        </h3>
                        <div className="space-y-4">
                          <div>
                            <label className="text-sm text-amber-200/60 block mb-2">QUG Amount</label>
                            <div className="relative">
                              <input
                                type="number"
                                value={mintAmount}
                                onChange={(e) => setMintAmount(e.target.value)}
                                placeholder="0.00"
                                className="w-full p-3 rounded-lg bg-slate-800/50 border border-amber-500/20 text-amber-100 focus:border-amber-500 focus:outline-none"
                              />
                              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-amber-200/60">QUG</span>
                            </div>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-amber-200/60">You'll receive</span>
                            <span className="text-amber-100">
                              {mintAmount ? (parseFloat(mintAmount) / selectedFund.navPerShare).toFixed(4) : '0'} {selectedFund.symbol}
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-amber-200/60">Fee (0.1%)</span>
                            <span className="text-amber-100">
                              {mintAmount ? (parseFloat(mintAmount) * 0.001).toFixed(4) : '0'} QUG
                            </span>
                          </div>
                          <motion.button
                            className="w-full py-3 rounded-lg font-semibold text-slate-900"
                            style={{
                              background: 'linear-gradient(135deg, #10B981, #34D399)',
                            }}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            Mint Shares
                          </motion.button>
                        </div>
                      </div>

                      {/* Redeem Section */}
                      <div className="p-5 rounded-xl bg-red-500/10 border border-red-500/30">
                        <h3 className="text-lg font-semibold text-red-400 flex items-center gap-2 mb-4">
                          <Minus className="w-5 h-5" />
                          Redeem Shares
                        </h3>
                        <div className="space-y-4">
                          <div>
                            <label className="text-sm text-amber-200/60 block mb-2">Shares Amount</label>
                            <div className="relative">
                              <input
                                type="number"
                                value={redeemAmount}
                                onChange={(e) => setRedeemAmount(e.target.value)}
                                placeholder="0.00"
                                className="w-full p-3 rounded-lg bg-slate-800/50 border border-amber-500/20 text-amber-100 focus:border-amber-500 focus:outline-none"
                              />
                              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-amber-200/60">{selectedFund.symbol}</span>
                            </div>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-amber-200/60">You'll receive</span>
                            <span className="text-amber-100">
                              {redeemAmount ? (parseFloat(redeemAmount) * selectedFund.navPerShare).toFixed(4) : '0'} QUG
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-amber-200/60">Fee (0.1%)</span>
                            <span className="text-amber-100">
                              {redeemAmount ? (parseFloat(redeemAmount) * selectedFund.navPerShare * 0.001).toFixed(4) : '0'} QUG
                            </span>
                          </div>
                          <motion.button
                            className="w-full py-3 rounded-lg font-semibold text-white"
                            style={{
                              background: 'linear-gradient(135deg, #EF4444, #F87171)',
                            }}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            disabled={selectedFund.myShares <= 0}
                          >
                            Redeem Shares
                          </motion.button>
                        </div>
                      </div>
                    </div>

                    {/* My Holdings */}
                    {selectedFund.myShares > 0 && (
                      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30">
                        <h4 className="font-semibold text-amber-100 mb-2">Your Position</h4>
                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <p className="text-sm text-amber-200/60">Shares Held</p>
                            <p className="text-lg font-semibold text-amber-100">{selectedFund.myShares.toFixed(4)}</p>
                          </div>
                          <div>
                            <p className="text-sm text-amber-200/60">Current Value</p>
                            <p className="text-lg font-semibold text-amber-100">{formatCurrency(selectedFund.myValue)}</p>
                          </div>
                          <div>
                            <p className="text-sm text-amber-200/60">% of Fund</p>
                            <p className="text-lg font-semibold text-amber-100">
                              {((selectedFund.myShares / selectedFund.totalSupply) * 100).toFixed(4)}%
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {activeTab === 'governance' && (
                  <motion.div
                    key="governance"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="space-y-4"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-amber-100">Active Proposals</h3>
                      <button className="text-sm text-amber-400 hover:text-amber-300 flex items-center gap-1">
                        <Plus className="w-4 h-4" />
                        Create Proposal
                      </button>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-800/50 border border-amber-500/20">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <p className="font-medium text-amber-100">Proposal #1: Add QSUI to Index</p>
                          <p className="text-sm text-amber-200/60">Add Quantum Sui at 5% target weight</p>
                        </div>
                        <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400">
                          Active
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-amber-200/60">For</span>
                          <span className="text-green-400">65.4%</span>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full bg-green-500 w-[65%] rounded-full" />
                        </div>
                        <div className="flex justify-between text-xs text-amber-200/60">
                          <span>Quorum: 10% (Reached)</span>
                          <span>Ends in 3 days</span>
                        </div>
                      </div>
                      <div className="flex gap-2 mt-4">
                        <button className="flex-1 py-2 rounded-lg bg-green-500/20 text-green-400 font-medium hover:bg-green-500/30">
                          Vote For
                        </button>
                        <button className="flex-1 py-2 rounded-lg bg-red-500/20 text-red-400 font-medium hover:bg-red-500/30">
                          Vote Against
                        </button>
                      </div>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-800/30 border border-amber-500/10 opacity-60">
                      <div className="flex items-center gap-2 text-amber-200/60">
                        <History className="w-5 h-5" />
                        <span>No historical proposals yet</span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
