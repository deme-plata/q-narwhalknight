import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Code, Coins, Building, Vote, Lock, ArrowRight, Sparkles, CheckCircle, FileCode, Settings, Flame, Zap, Users, PauseCircle, PlayCircle, RefreshCw, Upload, Send, History, BarChart3, TrendingUp, Activity, Clock, ArrowUpRight, ArrowDownRight, Gift, Percent, PieChart, Landmark, Gem, Leaf, Palette, FileText, Package, Shield, DollarSign, Search, ArrowUpDown, Filter, X, Wallet, HelpCircle } from 'lucide-react';
import { generateAuthHeader, walletSession } from '../services/walletAuth';
import RwaPortfolioTab from './RwaPortfolioTab';
import {
  contractTemplates,
  rwaFieldsByTemplate,
  contractTypeMap,
  feeMultipliers,
  categories,
  calculateDynamicGas,
  TARGET_FEE_USD,
  type ContractCategory,
  type ContractTemplate,
  type RwaField,
} from './rwaTemplateData';

// ContractCategory is now imported from rwaTemplateData
type _ContractCategoryAlias = ContractCategory; // Keep for backward compatibility
type DeploymentStep = 'select' | 'basics' | 'features' | 'review' | 'deploying' | 'success';
type ContractTab = 'control' | 'events' | 'stats' | 'social';

// Event types for contract history
interface ContractEvent {
  id: string;
  type: 'mint' | 'burn' | 'transfer' | 'airdrop' | 'pause' | 'unpause' | 'stake' | 'unstake' | 'reflection';
  amount?: string;
  from?: string;
  to?: string;
  recipients?: number;
  timestamp: Date;
  txHash: string;
}

// Token stats interface
interface TokenStats {
  totalSupply: string;
  circulatingSupply: string;
  burnedTokens: string;
  holders: number;
  totalTransfers: number;
  totalMinted: string;
  totalBurned: string;
  totalAirdropped: string;
  stakingAPY?: string;
  totalStaked?: string;
  reflectionRate?: string;
  totalReflections?: string;
  loading?: boolean;
}

// v2.4.8: Social media profile for token creators
interface SocialMediaProfile {
  twitter?: string;
  discord?: string;
  telegram?: string;
  website?: string;
  github?: string;
  medium?: string;
  description?: string;
}

// v2.4.8: Creator/Developer history for trust scoring
interface CreatorHistory {
  address: string;
  tokensCreated: {
    symbol: string;
    name: string;
    deployedAt: Date;
    marketCap?: number;
    status: 'active' | 'abandoned' | 'rugged';
    percentChange?: number;
  }[];
  totalTokensCreated: number;
  successfulTokens: number;
  ruggedTokens: number;
  trustScore: number; // 0-100
}

interface DeployedContract {
  address: string;
  name: string;
  symbol: string;
  type: string;
  deployedAt: Date;
  features: {
    mintable?: boolean;
    burnable?: boolean;
    reflection?: boolean;
    staking?: boolean;
    governance?: boolean;
    pausable?: boolean;
    upgradeable?: boolean;
    airdrop?: boolean;
    // RWA-specific features
    kyc_required?: boolean;
    accredited_only?: boolean;
    dividend_enabled?: boolean;
    transfer_restrictions?: boolean;
    voting_rights?: boolean;
    callable?: boolean;
    convertible?: boolean;
    delivery_option?: boolean;
    insurance_enabled?: boolean;
    retirement_enabled?: boolean;
    offset_tracking?: boolean;
    provenance_verified?: boolean;
    redemption_enabled?: boolean;
    sublicensing_allowed?: boolean;
    [key: string]: boolean | undefined; // Allow dynamic RWA feature keys
  };
  isPaused?: boolean;
  logoUrl?: string; // libp2p IPFS CID for logo
  logoDataUrl?: string; // Base64 data URL for display (temporary until uploaded)
  abaBalance?: string; // User's balance of this token
  totalSupply?: string; // v1.4.9: Total supply of the token (for event history)
  decimals?: number; // v1.4.9: Token decimals for balance conversion
  owner?: string; // v2.4.8: Contract owner/creator address
  socialMedia?: SocialMediaProfile; // v2.4.8: Social links
  deploymentParams?: Record<string, string | boolean>; // RWA deployment parameters
}

// ContractTemplate, ContractCategory, RwaField interfaces and constants
// are all imported from ./rwaTemplateData (contractTemplates, rwaFieldsByTemplate,
// contractTypeMap, feeMultipliers, categories, calculateDynamicGas, TARGET_FEE_USD)

// ─── Tooltip helpers for RWA controls ───────────────
function RwaTooltip({ text, children, position = 'top' }: { text: string; children: React.ReactNode; position?: 'top' | 'bottom' | 'left' | 'right' }) {
  const [show, setShow] = useState(false);
  const pos: Record<string, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  return (
    <span className="relative inline-flex items-center" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span className={`absolute z-[100] ${pos[position] || pos.top} pointer-events-none`}>
          <span className="block bg-gray-800 text-gray-200 text-[11px] leading-[1.4] px-3 py-2 rounded-lg shadow-xl shadow-black/40 border border-gray-700/50 max-w-[260px] min-w-[140px] whitespace-normal font-normal">
            {text}
          </span>
        </span>
      )}
    </span>
  );
}

function RwaInfoTip({ text, position }: { text: string; position?: 'top' | 'bottom' | 'right' }) {
  return (
    <RwaTooltip text={text} position={position || 'top'}>
      <HelpCircle className="w-3.5 h-3.5 text-gray-500 hover:text-gray-300 transition-colors cursor-help ml-1 inline" />
    </RwaTooltip>
  );
}

export default function VittuaVMScreen() {
  const [vmView, setVmView] = useState<'deploy' | 'contracts' | 'portfolio'>('deploy');
  const [step, setStep] = useState<DeploymentStep>('select');
  const [selectedCategory, setSelectedCategory] = useState<ContractCategory>('tokens');
  const [selectedTemplate, setSelectedTemplate] = useState<ContractTemplate | null>(null);

  // Oracle price state for dynamic gas calculation
  const [qugPriceUsd, setQugPriceUsd] = useState<number>(0.01);

  // Fetch QUG price from oracle
  React.useEffect(() => {
    const fetchOraclePrice = async () => {
      try {
        const response = await fetch('/api/v1/oracle/price');
        const data = await response.json();
        if (data.success && data.data) {
          setQugPriceUsd(data.data.price_usd);
          console.log('🔮 Oracle price updated:', data.data.price_usd, 'USD per QUG');
        }
      } catch (error) {
        console.error('Failed to fetch oracle price:', error);
      }
    };

    // Fetch price on mount and every 30 seconds
    fetchOraclePrice();
    const interval = setInterval(fetchOraclePrice, 30000);
    return () => clearInterval(interval);
  }, []);

  // Contract basics
  const [contractName, setContractName] = useState('');
  const [contractDescription, setContractDescription] = useState(''); // v4.0.1: Description for contract/token
  const [tokenSymbol, setTokenSymbol] = useState('');
  const [initialSupply, setInitialSupply] = useState('1000000');
  const [tokenDecimals, setTokenDecimals] = useState(8); // v3.2.18-beta: User-configurable decimals
  const [logoFile, setLogoFile] = useState<File | null>(null);
  const [logoPreview, setLogoPreview] = useState<string>('');
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // Advanced features
  const [features, setFeatures] = useState({
    mintable: true,
    burnable: true,
    reflection: false,
    staking: true,
    governance: false,
    pausable: true,
    upgradeable: false,
    airdrop: true
  });

  // RWA-specific parameters (dynamic per template)
  const [rwaParams, setRwaParams] = useState<Record<string, string | boolean>>({});

  const updateRwaParam = (key: string, value: string | boolean) => {
    setRwaParams(prev => ({ ...prev, [key]: value }));
  };

  // Check if selected template is an RWA type
  const isRwaTemplate = selectedTemplate?.category === 'rwa' && selectedTemplate?.id !== 'rwa-token';
  const currentRwaFields = selectedTemplate ? rwaFieldsByTemplate[selectedTemplate.id] || [] : [];

  // Deployed contracts - fetched from blockchain backend API
  const [deployedContracts, setDeployedContracts] = useState<DeployedContract[]>([]);
  const [loadingContracts, setLoadingContracts] = useState(true);
  const [lastDeployedAddress, setLastDeployedAddress] = useState<string>('');

  // Contract tab state - tracks which tab is active for each contract
  const [activeContractTabs, setActiveContractTabs] = useState<Record<string, ContractTab>>({});

  // Contract events - stores event history per contract
  const [contractEvents, setContractEvents] = useState<Record<string, ContractEvent[]>>({});

  // Contract stats - stores tokenomics per contract
  const [contractStats, setContractStats] = useState<Record<string, TokenStats>>({});

  // v2.4.8: Social media profiles per contract
  const [socialProfiles, setSocialProfiles] = useState<Record<string, SocialMediaProfile>>({});
  const [editingSocial, setEditingSocial] = useState<string | null>(null);
  const [socialFormData, setSocialFormData] = useState<SocialMediaProfile>({});
  const [savingSocial, setSavingSocial] = useState(false);

  // v2.4.8: Creator history for trust scoring (keyed by creator address)
  const [creatorHistories, setCreatorHistories] = useState<Record<string, CreatorHistory>>({});

  // Helper to get active tab for a contract (defaults to 'control')
  const getActiveTab = (contractAddress: string): ContractTab => {
    return activeContractTabs[contractAddress] || 'control';
  };

  // Helper to set active tab for a contract
  const setActiveTab = (contractAddress: string, tab: ContractTab) => {
    setActiveContractTabs(prev => ({ ...prev, [contractAddress]: tab }));
  };

  // Generate mock events for a contract (in real implementation, fetch from API)
  // v1.4.10: Fetch events from backend API
  const fetchContractEvents = async (contract: DeployedContract) => {
    try {
      // Strip qnk prefix for API call
      const contractAddr = contract.address.startsWith('qnk')
        ? contract.address.slice(3)
        : contract.address;

      const response = await fetch(`/api/v1/contracts/events/${contractAddr}`);
      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data?.events) {
          // Convert backend events to frontend format
          const apiEvents: ContractEvent[] = result.data.events.map((e: any) => ({
            id: e.id,
            type: e.event_type as ContractEvent['type'],
            amount: e.amount,
            from: e.from,
            to: e.to,
            recipients: e.recipients,
            timestamp: new Date(e.timestamp * 1000),
            txHash: e.tx_hash
          }));

          // v3.2.15-beta: If no events from API, use CURRENT balance (abaBalance) for fallback
          // Prefer abaBalance if available and non-zero, otherwise fall back to totalSupply
          if (apiEvents.length === 0) {
            const abaVal = contract.abaBalance && contract.abaBalance !== '0' ? contract.abaBalance : null;
            const totalVal = contract.totalSupply && contract.totalSupply !== '0' ? contract.totalSupply : null;
            const eventAmount = abaVal || totalVal || '0';
            apiEvents.push({
              id: '1',
              type: 'mint',
              amount: eventAmount,
              to: localStorage.getItem('walletAddress') || '',
              timestamp: contract.deployedAt,
              txHash: `0x${contract.address.slice(3, 11)}...initial`
            });
          }

          setContractEvents(prev => ({ ...prev, [contract.address]: apiEvents }));
          return apiEvents;
        }
      }
    } catch (error) {
      console.error('Failed to fetch contract events:', error);
    }

    // v3.2.15-beta: Fallback uses CURRENT balance (abaBalance) if available
    // Prefer abaBalance if non-zero, otherwise fall back to totalSupply
    const abaVal = contract.abaBalance && contract.abaBalance !== '0' ? contract.abaBalance : null;
    const totalVal = contract.totalSupply && contract.totalSupply !== '0' ? contract.totalSupply : null;
    const eventAmount = abaVal || totalVal || '0';
    const fallbackEvents: ContractEvent[] = [
      {
        id: '1',
        type: 'mint',
        amount: eventAmount,
        to: localStorage.getItem('walletAddress') || '',
        timestamp: contract.deployedAt,
        txHash: `0x${contract.address.slice(3, 11)}...initial`
      }
    ];
    setContractEvents(prev => ({ ...prev, [contract.address]: fallbackEvents }));
    return fallbackEvents;
  };

  const getContractEvents = (contract: DeployedContract): ContractEvent[] => {
    if (contractEvents[contract.address]) {
      return contractEvents[contract.address];
    }

    // v1.4.10: Trigger async fetch and return empty for now
    // The component will re-render when events are fetched
    fetchContractEvents(contract);

    return [];
  };

  // v3.2.15-beta: Format display amounts (user-entered values) with comma separators
  // This is for display amounts only - no decimal conversion needed
  // MUST be defined before getContractStats which uses it
  const formatDisplayAmount = (amount: string | undefined): string => {
    if (!amount) return '0';
    // Remove any existing commas and handle decimal numbers
    const cleaned = amount.replace(/,/g, '');
    const parts = cleaned.split('.');
    const intPart = parts[0] || '0';
    const decPart = parts[1];
    // Add comma separators to integer part
    const formattedInt = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return decPart ? `${formattedInt}.${decPart}` : formattedInt;
  };

  // v2.4.8: Fetch real stats from API
  const getContractStats = (contract: DeployedContract): TokenStats => {
    if (contractStats[contract.address]) {
      return contractStats[contract.address];
    }

    // Return loading state while fetching
    // v3.2.15-beta: Prefer abaBalance (current balance) if available and non-zero
    // Fall back to totalSupply if abaBalance not yet loaded
    const abaVal = contract.abaBalance && contract.abaBalance !== '0' ? contract.abaBalance : null;
    const totalVal = contract.totalSupply && contract.totalSupply !== '0' ? contract.totalSupply : null;
    const rawSupply = abaVal || totalVal || '0';
    const supplyDisplay = formatDisplayAmount(rawSupply);
    const loadingStats: TokenStats = {
      totalSupply: supplyDisplay,
      circulatingSupply: supplyDisplay,
      burnedTokens: '0',
      holders: 0,
      totalTransfers: 0,
      totalMinted: supplyDisplay,
      totalBurned: '0',
      totalAirdropped: '0',
      stakingAPY: contract.features.staking ? '12.5%' : undefined,
      totalStaked: contract.features.staking ? '0' : undefined,
      reflectionRate: contract.features.reflection ? '2%' : undefined,
      totalReflections: contract.features.reflection ? '0' : undefined,
      loading: true,
    };

    // Set loading state immediately
    setContractStats(prev => ({ ...prev, [contract.address]: loadingStats }));

    // Fetch real stats from API
    fetchContractStats(contract);

    return loadingStats;
  };

  // v2.4.8: Fetch token stats from backend API
  // v3.2.14-beta: Helper function to format large numbers using BigInt (prevents precision loss)
  const formatBigIntValue = (value: any, decimals: number = 8): string => {
    if (value === undefined || value === null) return '0';
    const valStr = typeof value === 'string' ? value : String(value);
    try {
      const bigVal = BigInt(valStr);
      const divisor = BigInt(10) ** BigInt(decimals);
      const wholePart = bigVal / divisor;
      const remainder = bigVal % divisor;
      if (remainder > 0n) {
        const fracStr = remainder.toString().padStart(decimals, '0');
        const trimmedFrac = fracStr.replace(/0+$/, '').slice(0, 2);
        return trimmedFrac ? `${wholePart.toLocaleString()}.${trimmedFrac}` : wholePart.toLocaleString();
      }
      return wholePart.toLocaleString();
    } catch {
      // Fallback for non-BigInt values
      const num = Number(valStr);
      return isNaN(num) ? '0' : num.toLocaleString();
    }
  };

  const fetchContractStats = async (contract: DeployedContract) => {
    try {
      const contractAddr = contract.address.startsWith('qnk')
        ? contract.address.slice(3)
        : contract.address;

      const response = await fetch(`/api/v1/contracts/${contractAddr}/token-stats`);
      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          const data = result.data;
          const decimals = contract.decimals || 8;

          // v3.2.15-beta: Use abaBalance (current user balance) as authoritative source
          // Prefer abaBalance if available and non-zero, otherwise fall back to API data
          const abaVal = contract.abaBalance && contract.abaBalance !== '0' ? contract.abaBalance : null;
          const currentBalance = abaVal || formatBigIntValue(data.total_supply, decimals);

          const stats: TokenStats = {
            totalSupply: currentBalance, // Use current balance as supply
            circulatingSupply: currentBalance,
            burnedTokens: formatBigIntValue(data.total_burned || '0', decimals),
            holders: data.holder_count || 0,
            totalTransfers: 0, // TODO: Add to API
            totalMinted: currentBalance, // Use current balance
            totalBurned: formatBigIntValue(data.total_burned || '0', decimals),
            totalAirdropped: '0', // TODO: Track airdrops
            stakingAPY: contract.features.staking ? '12.5%' : undefined,
            totalStaked: formatBigIntValue(data.total_staked || '0', decimals),
            reflectionRate: data.fee_config?.reflection_fee ? `${data.fee_config.reflection_fee}%` : undefined,
            totalReflections: formatBigIntValue(data.total_reflected || '0', decimals),
            loading: false,
          };
          setContractStats(prev => ({ ...prev, [contract.address]: stats }));
        }
      }
    } catch (error) {
      console.error('Failed to fetch token stats:', error);
    }
  };

  // v2.4.8: Fetch creator history for trust scoring
  const fetchCreatorHistory = async (creatorAddress: string) => {
    if (creatorHistories[creatorAddress]) return;

    try {
      const response = await fetch(`/api/v1/contracts/user/${creatorAddress}/contracts`);
      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          const contracts = result.data.contracts || [];
          const history: CreatorHistory = {
            address: creatorAddress,
            tokensCreated: contracts.map((c: any) => ({
              symbol: c.symbol || 'TOKEN',
              name: c.name || 'Unknown',
              deployedAt: new Date(c.deployed_at * 1000),
              status: 'active' as const,
            })),
            totalTokensCreated: contracts.length,
            successfulTokens: contracts.length, // All active for now
            ruggedTokens: 0,
            trustScore: Math.min(100, 50 + contracts.length * 10), // Base score + per token
          };
          setCreatorHistories(prev => ({ ...prev, [creatorAddress]: history }));
        }
      }
    } catch (error) {
      console.error('Failed to fetch creator history:', error);
    }
  };

  // v2.4.8: Save social media profile to backend (persisted + synced across nodes)
  const handleSaveSocial = async (contract: DeployedContract) => {
    setSavingSocial(true);
    try {
      const contractAddr = contract.address.startsWith('qnk')
        ? contract.address.slice(3)
        : contract.address;

      const walletAddress = localStorage.getItem('walletAddress') || '';

      const response = await fetch(`/api/v1/contracts/${contractAddr}/social`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...socialFormData,
          owner_address: walletAddress,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          setSocialProfiles(prev => ({ ...prev, [contract.address]: socialFormData }));
          setEditingSocial(null);
          console.log('✅ Social profile saved and synced for', contract.symbol);
        }
      } else {
        console.error('Failed to save social profile');
      }
    } catch (error) {
      console.error('Error saving social profile:', error);
    } finally {
      setSavingSocial(false);
    }
  };

  // v2.4.8: Load social profiles from backend API (decentralized across nodes)
  React.useEffect(() => {
    const loadSocialProfiles = async () => {
      for (const contract of deployedContracts) {
        try {
          const contractAddr = contract.address.startsWith('qnk')
            ? contract.address.slice(3)
            : contract.address;

          const response = await fetch(`/api/v1/contracts/${contractAddr}/social`);
          if (response.ok) {
            const result = await response.json();
            if (result.success && result.data) {
              const profile: SocialMediaProfile = {
                twitter: result.data.twitter,
                discord: result.data.discord,
                telegram: result.data.telegram,
                website: result.data.website,
                github: result.data.github,
                medium: result.data.medium,
                description: result.data.description,
              };
              // Only update if there's actual data
              if (Object.values(profile).some(v => v)) {
                setSocialProfiles(prev => ({ ...prev, [contract.address]: profile }));
              }
            }
          }
        } catch (error) {
          console.error('Error loading social profile for', contract.symbol, error);
        }
      }
    };

    if (deployedContracts.length > 0) {
      loadSocialProfiles();
    }
  }, [deployedContracts]);

  // Load contracts from localStorage on mount for instant display
  React.useEffect(() => {
    const walletAddress = localStorage.getItem('walletAddress');
    if (!walletAddress) return;

    // v1.4.9: Added cache version to invalidate old data missing totalSupply/decimals
    const CACHE_VERSION = 'v3';
    const cacheKey = `deployedContracts_${CACHE_VERSION}_${walletAddress}`;
    const cached = localStorage.getItem(cacheKey);

    // Clear old cache keys without version
    const oldCacheKey = `deployedContracts_${walletAddress}`;
    localStorage.removeItem(oldCacheKey);

    if (cached) {
      try {
        const contracts = JSON.parse(cached);
        // Convert date strings back to Date objects
        const parsedContracts = contracts.map((c: any) => ({
          ...c,
          deployedAt: new Date(c.deployedAt)
        }));
        setDeployedContracts(parsedContracts);
        console.log('📦 Loaded', parsedContracts.length, 'contracts from cache');
      } catch (error) {
        console.error('Failed to parse cached contracts:', error);
      }
    }
  }, []);

  // Fetch deployed contracts from backend API based on wallet address
  React.useEffect(() => {
    const fetchDeployedContracts = async () => {
      try {
        setLoadingContracts(true);

        // Get wallet address
        const walletAddress = localStorage.getItem('walletAddress');
        if (!walletAddress) {
          console.log('No wallet address found - skipping contract fetch');
          setLoadingContracts(false);
          return;
        }

        console.log('📡 Fetching deployed contracts for wallet:', walletAddress);

        // Fetch contracts from backend API
        // v1.4.9: Fixed path - was missing /contracts at the end
        const response = await fetch(`/api/v1/contracts/user/${walletAddress}/contracts`);

        if (!response.ok) {
          console.warn('Failed to fetch contracts:', response.status);
          setLoadingContracts(false);
          return;
        }

        const result = await response.json();

        if (result.success && result.data) {
          // Map backend contract data to frontend format
          const contractsWithoutBalances: DeployedContract[] = result.data.map((c: any) => {
            // v3.2.14-beta: Fix Total Supply display with proper BigInt decimal handling
            const rawSupply = c.total_supply || c.initial_supply || '0';
            const supplyStr = typeof rawSupply === 'string' ? rawSupply : String(rawSupply);
            const decimals = c.decimals || 8;

            // v3.2.14-beta: Use BigInt division to convert base units to display units
            // This prevents JavaScript Number precision loss for large integers
            const supplyBigInt = BigInt(supplyStr);
            const divisorBigInt = BigInt(10) ** BigInt(decimals);

            // Integer division for whole part
            const wholePart = supplyBigInt / divisorBigInt;
            const remainder = supplyBigInt % divisorBigInt;

            // Format with commas and optional decimals
            let formattedSupply: string;
            if (remainder > 0n) {
              const fracStr = remainder.toString().padStart(decimals, '0');
              const trimmedFrac = fracStr.replace(/0+$/, '').slice(0, 2); // Max 2 decimal places for supply
              formattedSupply = trimmedFrac ? `${wholePart.toLocaleString()}.${trimmedFrac}` : wholePart.toLocaleString();
            } else {
              formattedSupply = wholePart.toLocaleString();
            }

            return {
              address: c.address, // Already has qnk prefix from backend
              name: c.name,
              symbol: c.symbol || 'N/A',
              type: c.contract_type,
              deployedAt: new Date(c.deployed_at * 1000), // Convert Unix timestamp
              features: c.features || {},
              isPaused: false,
              totalSupply: formattedSupply, // v3.2.14-beta: Fixed - now in display units
              decimals: decimals,
              deploymentParams: c.deployment_params || {}, // v4.0.3: RWA config params
            };
          });

          // Fetch token balance for each contract (user's balance OF each token)
          const contractsWithBalances = await Promise.all(
            contractsWithoutBalances.map(async (contract) => {
              try {
                // Fetch user's balance of this token from API
                const balanceResponse = await fetch(`/api/v1/contracts/${contract.address}/balance/${walletAddress}`);
                if (balanceResponse.ok) {
                  const balanceResult = await balanceResponse.json();
                  if (balanceResult.success && balanceResult.data) {
                    // v3.2.14-beta: Fix BigInt precision loss bug
                    // Backend returns balance in BASE UNITS - use BigInt to prevent precision loss
                    const rawBalance = balanceResult.data.balance || '0';
                    const balanceStr = typeof rawBalance === 'string' ? rawBalance : String(rawBalance);

                    // Get decimals (default to 8 like QUG/QUGUSD)
                    const decimals = contract.decimals || 8;

                    // v3.2.14-beta: Use BigInt for large number division (prevents JS Number precision loss)
                    // JavaScript Number can only safely represent integers up to 2^53
                    // Our balances can be up to 10^37 which would lose precision
                    const balanceBigInt = BigInt(balanceStr);
                    const divisorBigInt = BigInt(10) ** BigInt(decimals);

                    // Integer division for whole part
                    const wholePart = balanceBigInt / divisorBigInt;
                    // Remainder for fractional part
                    const remainder = balanceBigInt % divisorBigInt;

                    // Format whole part with commas
                    const wholeStr = wholePart.toLocaleString();

                    // Format fractional part (pad with leading zeros, trim trailing zeros)
                    let formattedBalance: string;
                    if (remainder > 0n) {
                      const fracStr = remainder.toString().padStart(decimals, '0');
                      const trimmedFrac = fracStr.replace(/0+$/, ''); // Remove trailing zeros
                      formattedBalance = trimmedFrac ? `${wholeStr}.${trimmedFrac.slice(0, 4)}` : wholeStr;
                    } else {
                      formattedBalance = wholeStr;
                    }

                    console.log(`✅ Fetched balance for ${contract.symbol}: ${rawBalance} base units → ${formattedBalance} display (decimals: ${decimals})`);
                    return { ...contract, abaBalance: formattedBalance };
                  }
                }
                console.warn(`⚠️ Failed to fetch balance for contract ${contract.address} - response not ok or no data`);
              } catch (error) {
                console.warn(`❌ Failed to fetch balance for contract ${contract.address}:`, error);
              }
              return { ...contract, abaBalance: '0' };
            })
          );

          console.log('✅ Loaded', contractsWithBalances.length, 'deployed contracts from blockchain with ABA balances');
          setDeployedContracts(contractsWithBalances);

          // Cache contracts in localStorage for instant display on next mount
          // v1.4.9: Use versioned cache key
          const CACHE_VERSION = 'v2';
          const cacheKey = `deployedContracts_${CACHE_VERSION}_${walletAddress}`;
          localStorage.setItem(cacheKey, JSON.stringify(contractsWithBalances));
          console.log('💾 Cached contracts to localStorage (v2)');
        }
      } catch (error) {
        console.error('Failed to fetch deployed contracts:', error);
      } finally {
        setLoadingContracts(false);
      }
    };

    fetchDeployedContracts();

    // Refresh contracts every 30 seconds
    const interval = setInterval(fetchDeployedContracts, 30000);
    return () => clearInterval(interval);
  }, [step]); // Re-fetch when returning to 'select' step

  // Deployed contracts search, sort, and filter state
  const [contractSearch, setContractSearch] = useState('');
  const [contractSort, setContractSort] = useState<'name' | 'date' | 'type' | 'symbol'>('date');
  const [contractSortDir, setContractSortDir] = useState<'asc' | 'desc'>('desc');
  const [contractTypeFilter, setContractTypeFilter] = useState<string>('all');

  const filteredDeployedContracts = React.useMemo(() => {
    let filtered = [...deployedContracts];

    // Search filter
    if (contractSearch.trim()) {
      const q = contractSearch.toLowerCase();
      filtered = filtered.filter(c =>
        c.name.toLowerCase().includes(q) ||
        c.symbol.toLowerCase().includes(q) ||
        c.type.toLowerCase().includes(q) ||
        c.address.toLowerCase().includes(q)
      );
    }

    // Type filter
    if (contractTypeFilter !== 'all') {
      filtered = filtered.filter(c => {
        const ct = c.type.toLowerCase();
        if (contractTypeFilter === 'rwa') return ['realestate', 'equity', 'fixedincome', 'commodity', 'carbon', 'art', 'collectible', 'ip', 'royalt', 'physical', 'rwa'].some(t => ct.includes(t));
        if (contractTypeFilter === 'token') return ct.includes('token') && !['realestate', 'equity', 'fixedincome', 'commodity', 'carbon', 'art', 'collectible', 'ip', 'royalt', 'physical', 'rwa'].some(t => ct.includes(t));
        if (contractTypeFilter === 'governance') return ct.includes('governance') || ct.includes('dao');
        if (contractTypeFilter === 'defi') return ct.includes('dex') || ct.includes('defi');
        return true;
      });
    }

    // Sort
    filtered.sort((a, b) => {
      let cmp = 0;
      switch (contractSort) {
        case 'name': cmp = a.name.localeCompare(b.name); break;
        case 'symbol': cmp = a.symbol.localeCompare(b.symbol); break;
        case 'type': cmp = a.type.localeCompare(b.type); break;
        case 'date': cmp = (a.deployedAt?.getTime() || 0) - (b.deployedAt?.getTime() || 0); break;
      }
      return contractSortDir === 'desc' ? -cmp : cmp;
    });

    return filtered;
  }, [deployedContracts, contractSearch, contractSort, contractSortDir, contractTypeFilter]);

  // Contract control states
  const [mintAmount, setMintAmount] = useState('');
  const [burnAmount, setBurnAmount] = useState('');
  const [reflectionRate, setReflectionRate] = useState('2');
  const [airdropAddresses, setAirdropAddresses] = useState('');
  const [airdropAmount, setAirdropAmount] = useState('');

  // v4.0.3: RWA action state - inline feedback instead of browser alerts
  const [rwaActionStatus, setRwaActionStatus] = useState<Record<string, { loading: boolean; message: string; success: boolean }>>({});
  const [rwaInputs, setRwaInputs] = useState<Record<string, string>>({});

  const getRwaInput = (key: string) => rwaInputs[key] || '';
  const setRwaInput = (key: string, value: string) => setRwaInputs(prev => ({ ...prev, [key]: value }));
  const getRwaStatus = (contractAddr: string, action: string) => rwaActionStatus[`${contractAddr}:${action}`];
  const setRwaStatus = (contractAddr: string, action: string, status: { loading: boolean; message: string; success: boolean }) => {
    setRwaActionStatus(prev => ({ ...prev, [`${contractAddr}:${action}`]: status }));
    // Auto-clear success/error messages after 5 seconds
    if (!status.loading) {
      setTimeout(() => {
        setRwaActionStatus(prev => {
          const next = { ...prev };
          delete next[`${contractAddr}:${action}`];
          return next;
        });
      }, 5000);
    }
  };

  const toggleFeature = (feature: keyof typeof features) => {
    setFeatures(prev => ({ ...prev, [feature]: !prev[feature] }));
  };

  const filteredTemplates = contractTemplates.filter(t => t.category === selectedCategory);

  const handleSelectTemplate = (template: ContractTemplate) => {
    setSelectedTemplate(template);
    setRwaParams({}); // Reset RWA params when switching templates
    setContractDescription(''); // Reset description when switching templates
    setStep('basics');
  };

  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/png')) {
      alert('Please upload a PNG image');
      return;
    }

    // Validate file size (max 500KB)
    if (file.size > 500 * 1024) {
      alert('Logo must be smaller than 500KB');
      return;
    }

    setLogoFile(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setLogoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const uploadLogoToLibp2p = async (file: File): Promise<string> => {
    try {
      // Convert image to bytes for libp2p upload
      const bytes = await file.arrayBuffer();
      const uint8Array = new Uint8Array(bytes);

      // Upload to libp2p/IPFS via our API
      const response = await fetch('/api/v1/ipfs/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-Filename': file.name,
        },
        body: uint8Array,
      });

      if (!response.ok) {
        throw new Error('Failed to upload logo to libp2p');
      }

      const result = await response.json();
      return result.data.cid; // Return IPFS CID
    } catch (error) {
      console.error('Logo upload failed:', error);
      throw error;
    }
  };

  const handleDeploy = async () => {
    setStep('deploying');

    try {
      // Get wallet ID from localStorage - try multiple approaches for compatibility
      let walletId: string | null = null;

      // First, try to get from currentWallet JSON object
      const walletData = localStorage.getItem('currentWallet');
      if (walletData) {
        try {
          const wallet = JSON.parse(walletData);
          if (wallet && wallet.id) {
            walletId = wallet.id;
          }
        } catch (e) {
          console.warn('Failed to parse currentWallet:', e);
        }
      }

      // Fallback: try to get from separate walletId key
      if (!walletId) {
        walletId = localStorage.getItem('walletId');
      }

      // Fallback: derive from wallet address if needed
      if (!walletId) {
        const walletAddress = localStorage.getItem('walletAddress');
        if (walletAddress) {
          // Use wallet address without 'qnk' prefix as ID
          walletId = walletAddress.replace('qnk', '');
        }
      }

      // Validate we have a wallet ID
      if (!walletId) {
        throw new Error('No wallet found. Please create a wallet first or reload the page.');
      }

      console.log('💼 Wallet ID for signing:', walletId);
      console.log('💼 Wallet ID type:', typeof walletId);
      console.log('💼 Wallet ID length:', walletId.length);

      // Calculate gas cost dynamically based on USD target and oracle price
      // Template multipliers: secure-token=1x, advanced-token=2x, rwa-token=3x, governance=4x, private-dex=5x
      const feeMultiplier =
        selectedTemplate?.id === 'secure-token' ? 1 :
        selectedTemplate?.id === 'advanced-token' ? 2 :
        selectedTemplate?.id === 'rwa-token' ? 3 :
        selectedTemplate?.id === 'real-estate-token' ? 4 :
        selectedTemplate?.id === 'equity-token' ? 4 :
        selectedTemplate?.id === 'fixed-income-token' ? 3.5 :
        selectedTemplate?.id === 'commodity-token' ? 3.5 :
        selectedTemplate?.id === 'carbon-credit-token' ? 3 :
        selectedTemplate?.id === 'art-collectible-token' ? 3.5 :
        selectedTemplate?.id === 'ip-revenue-token' ? 3.5 :
        selectedTemplate?.id === 'physical-goods-token' ? 3.5 :
        selectedTemplate?.id === 'governance' ? 4 :
        selectedTemplate?.id === 'private-dex' ? 5 : 1;

      const targetUsdCost = TARGET_FEE_USD * feeMultiplier;
      const gasAmount = calculateDynamicGas(targetUsdCost, qugPriceUsd);
      const fee = Math.floor(gasAmount * 1_000_000_000); // Convert to base units (nanoQUG)

      console.log('🚀 Contract deployment:', {
        name: contractName,
        symbol: tokenSymbol,
        type: selectedTemplate?.name,
        gasEstimate: `${gasAmount} QUG`,
        fee: `${fee} QUG`,
        hasLogo: !!logoFile
      });

      // Upload logo to libp2p if provided
      let logoIpfsCid = '';
      if (logoFile) {
        console.log('📤 Uploading logo to libp2p/IPFS...');
        logoIpfsCid = await uploadLogoToLibp2p(logoFile);
        console.log('✅ Logo uploaded to IPFS:', logoIpfsCid);
      }

      // Create contract metadata JSON
      const metadata = {
        name: contractName,
        symbol: tokenSymbol,
        type: selectedTemplate?.id,
        template: selectedTemplate?.name,
        initialSupply,
        features: selectedTemplate?.id === 'advanced-token' ? features : {},
        rwaParameters: isRwaTemplate ? rwaParams : undefined,
        logoIpfsCid: logoIpfsCid || undefined,
        deployedAt: new Date().toISOString(),
      };

      // Encode metadata as UTF-8 bytes for transaction data field
      const metadataJson = JSON.stringify(metadata);
      const encoder = new TextEncoder();
      const metadataBytes = Array.from(encoder.encode(metadataJson));

      console.log('📝 Contract metadata:', metadata);
      console.log('📊 Metadata size:', metadataBytes.length, 'bytes');

      // Deploy contract via backend API (bypassing the zero-address transaction method)
      // Get wallet address for contract ownership
      const walletAddress = localStorage.getItem('walletAddress');
      if (!walletAddress) {
        throw new Error('No wallet address found - cannot determine contract owner');
      }

      console.log('🚀 Deploying contract via backend API...');
      console.log('👤 Contract owner:', walletAddress);
      console.log('📝 Contract type:', selectedTemplate?.id);

      // Map frontend template ID to backend contract type
      const contractTypeMap: Record<string, string> = {
        'secure-token': 'secure_token',
        'advanced-token': 'advanced_token',
        'rwa-token': 'rwa_token',
        'real-estate-token': 'real_estate_token',
        'equity-token': 'equity_token',
        'fixed-income-token': 'fixed_income_token',
        'commodity-token': 'commodity_token',
        'carbon-credit-token': 'carbon_credit_token',
        'art-collectible-token': 'art_collectible_token',
        'ip-revenue-token': 'ip_revenue_token',
        'physical-goods-token': 'physical_goods_token',
        'governance': 'governance',
        'private-dex': 'private_dex',
      };

      const backendContractType = contractTypeMap[selectedTemplate?.id || ''] || 'secure_token';

      // v3.2.18-beta: Convert initialSupply to BASE UNITS before sending to backend
      // User enters display units (e.g., "1000000"), we multiply by 10^decimals for storage
      // This ensures consistency: backend stores base units, frontend divides when displaying
      const decimals = tokenDecimals; // v3.2.18-beta: Use user-selected decimals
      let initialSupplyBaseUnits: string;
      try {
        // v3.2.19-beta: Use BigInt throughout to prevent precision loss
        // CRITICAL: 10 ** 18 exceeds Number.MAX_SAFE_INTEGER, causing precision loss
        // Must use BigInt(10) ** BigInt(decimals) instead of BigInt(10) ** BigInt(decimals)
        const displayAmount = initialSupply.replace(/,/g, ''); // Remove any commas
        const parts = displayAmount.split('.');
        const wholePart = parts[0] || '0';
        const fracPart = (parts[1] || '').slice(0, decimals).padEnd(decimals, '0');

        // FIXED: Use BigInt exponentiation to avoid Number precision loss
        const multiplier = BigInt(10) ** BigInt(decimals);
        const baseUnits = BigInt(wholePart) * multiplier + BigInt(fracPart || '0');
        initialSupplyBaseUnits = baseUnits.toString();
        console.log(`📊 Initial supply: ${initialSupply} display × 10^${decimals} = ${initialSupplyBaseUnits} base units`);
      } catch (e) {
        console.error('Failed to convert initial supply to base units:', e);
        initialSupplyBaseUnits = initialSupply; // Fallback to raw value
      }

      // Prepare deployment request
      const deploymentRequest = {
        contract_type: backendContractType,
        owner: walletAddress,
        parameters: {
          name: contractName,
          symbol: tokenSymbol,
          description: contractDescription || undefined, // v4.0.1: Include description
          initialSupply: initialSupplyBaseUnits, // v3.2.18-beta: Send BASE units
          decimals: tokenDecimals, // v3.2.18-beta: User-selected decimals
          logoIpfsCid: logoIpfsCid || undefined,
          ...features, // Include all feature flags
          ...(isRwaTemplate ? rwaParams : {}), // Include RWA-specific parameters
        },
        deployment_options: {
          test_deployment: false,
          auto_verify: true,
          enable_governance: features.governance || false,
          enable_upgrades: features.upgradeable || false,
          gas_limit: fee,
          deploy_with_proxy: false,
        },
      };

      console.log('📡 Sending deployment request:', deploymentRequest);

      // Generate wallet auth header for authenticated deployment
      const session = walletSession.getSession();
      if (!session) {
        throw new Error('Wallet session not found. Please log in again.');
      }
      const authHeader = await generateAuthHeader(
        session.privateKey,
        session.address,
        '/api/v1/contracts/deploy'
      );

      // Call the contract deployment API
      const deployResponse = await fetch('/api/v1/contracts/deploy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': authHeader,
        },
        body: JSON.stringify(deploymentRequest),
      });

      if (!deployResponse.ok) {
        let errorMsg = `Server returned ${deployResponse.status}`;
        try {
          const error = await deployResponse.json();
          errorMsg = error.error || errorMsg;
        } catch (_) {
          // Response may not be JSON
        }
        throw new Error(errorMsg);
      }

      const deployResult = await deployResponse.json();
      console.log('✅ Contract deployment response:', deployResult);

      // Validate deployment result
      if (!deployResult.success || !deployResult.data) {
        throw new Error(deployResult.error || 'Contract deployment failed');
      }

      // Extract the actual contract address from the response
      const contractAddress = deployResult.data.contract_address;

      if (!contractAddress) {
        throw new Error('Backend did not return a contract address');
      }

      console.log('🎉 Contract successfully deployed at:', contractAddress);

      // Save the deployed address for success screen display
      setLastDeployedAddress(contractAddress);

      // Add deployed contract to local state
      const newContract: DeployedContract = {
        address: contractAddress, // Use transaction hash as contract address
        name: contractName,
        symbol: tokenSymbol,
        type: selectedTemplate?.name || 'Token',
        deployedAt: new Date(),
        features: selectedTemplate?.id === 'advanced-token' ? features : {},
        isPaused: false,
        logoUrl: logoIpfsCid ? `ipfs://${logoIpfsCid}` : undefined,
        logoDataUrl: logoPreview || undefined,
        decimals: tokenDecimals, // v3.2.18-beta: Store user-selected decimals
        totalSupply: initialSupply, // v3.2.18-beta: Store initial supply in display units
        deploymentParams: isRwaTemplate ? rwaParams : undefined, // RWA deployment parameters
      };

      const updatedContracts = [...deployedContracts, newContract];
      setDeployedContracts(updatedContracts);

      // Update localStorage cache immediately (walletAddress already defined above)
      const cacheKey = `deployedContracts_${walletAddress}`;
      localStorage.setItem(cacheKey, JSON.stringify(updatedContracts));
      console.log('💾 Updated cache with newly deployed contract');

      // v4.0.1: Auto-save description to social profile if provided
      if (contractDescription.trim()) {
        try {
          await fetch(`/api/v1/contracts/${contractAddress}/social`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              description: contractDescription.trim(),
              owner_address: walletAddress,
            }),
          });
          setSocialProfiles(prev => ({
            ...prev,
            [contractAddress]: { description: contractDescription.trim() },
          }));
          console.log('📝 Auto-saved description to social profile');
        } catch (err) {
          console.warn('Failed to auto-save description:', err);
        }
      }

      setStep('success');

    } catch (error: any) {
      console.error('❌ Contract deployment failed:', error);
      alert(`Deployment failed: ${error.message}`);
      setStep('review'); // Go back to review step on error
    }
  };

  const handleMint = async (contract: DeployedContract) => {
    if (!mintAmount || mintAmount === '0') {
      alert('Please enter a valid amount to mint');
      return;
    }

    try {
      // v3.2.14-beta: Use BigInt to convert display units to base units (prevents precision loss)
      // This is critical for large token amounts like 1e30 which exceed Number precision
      const decimals = contract.decimals || 8;

      // Parse the input which may have decimals (e.g., "123.456")
      const parts = mintAmount.split('.');
      const wholePart = parts[0] || '0';
      const fracPart = (parts[1] || '').slice(0, decimals).padEnd(decimals, '0');

      // Combine whole and fractional parts as BigInt in base units
      const baseUnits = BigInt(wholePart) * BigInt(10) ** BigInt(decimals) + BigInt(fracPart);

      console.log(`🪙 Minting ${mintAmount} ${contract.symbol} (${baseUnits.toString()} base units) for contract ${contract.address}`);

      const response = await fetch('/api/v1/contracts/mint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contract.address,
          amount: baseUnits.toString(), // v1.4.10: Send base units, not display units
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to mint tokens');
      }

      const result = await response.json();
      console.log('✅ Mint successful:', result);

      // v1.4.10: Add mint event to event history
      const newEvent: ContractEvent = {
        id: `mint-${Date.now()}`,
        type: 'mint',
        amount: mintAmount, // Display units for UI
        to: localStorage.getItem('walletAddress') || '',
        timestamp: new Date(),
        txHash: result.data?.transaction_hash || `0x${Date.now().toString(16)}`
      };
      setContractEvents(prev => ({
        ...prev,
        [contract.address]: [newEvent, ...(prev[contract.address] || [])]
      }));

      alert(`Successfully minted ${mintAmount} ${contract.symbol}!`);
      setMintAmount('');
    } catch (error: any) {
      console.error('❌ Mint failed:', error);
      alert(`Mint failed: ${error.message}`);
    }
  };

  const handleBurn = async (contract: DeployedContract) => {
    if (!burnAmount || burnAmount === '0') {
      alert('Please enter a valid amount to burn');
      return;
    }

    try {
      // v3.2.14-beta: Use BigInt to convert display units to base units (prevents precision loss)
      // This is critical for large token amounts like 1e30 which exceed Number precision
      const decimals = contract.decimals || 8;

      // Parse the input which may have decimals (e.g., "123.456")
      const parts = burnAmount.split('.');
      const wholePart = parts[0] || '0';
      const fracPart = (parts[1] || '').slice(0, decimals).padEnd(decimals, '0');

      // Combine whole and fractional parts as BigInt in base units
      const baseUnits = BigInt(wholePart) * BigInt(10) ** BigInt(decimals) + BigInt(fracPart);

      console.log(`🔥 Burning ${burnAmount} ${contract.symbol} (${baseUnits.toString()} base units) from contract ${contract.address}`);

      const response = await fetch('/api/v1/contracts/burn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contract.address,
          amount: baseUnits.toString(), // v1.4.10: Send base units, not display units
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to burn tokens');
      }

      const result = await response.json();
      console.log('✅ Burn successful:', result);

      // v1.4.10: Add burn event to event history
      const newEvent: ContractEvent = {
        id: `burn-${Date.now()}`,
        type: 'burn',
        amount: burnAmount, // Display units for UI
        from: localStorage.getItem('walletAddress') || '',
        timestamp: new Date(),
        txHash: result.data?.transaction_hash || `0x${Date.now().toString(16)}`
      };
      setContractEvents(prev => ({
        ...prev,
        [contract.address]: [newEvent, ...(prev[contract.address] || [])]
      }));

      alert(`Successfully burned ${burnAmount} ${contract.symbol}!`);
      setBurnAmount('');
    } catch (error: any) {
      console.error('❌ Burn failed:', error);
      alert(`Burn failed: ${error.message}`);
    }
  };

  const handleAirdrop = async (contract: DeployedContract) => {
    if (!airdropAddresses || !airdropAmount || airdropAmount === '0') {
      alert('Please enter valid addresses and amount');
      return;
    }

    // Parse addresses (comma or newline separated)
    const addresses = airdropAddresses
      .split(/[\n,]+/)
      .map(addr => addr.trim())
      .filter(addr => addr.length > 0);

    if (addresses.length === 0) {
      alert('Please enter at least one address');
      return;
    }

    try {
      // v3.2.14-beta: Use BigInt to convert display units to base units (prevents precision loss)
      // This is critical for large token amounts like 1e30 which exceed Number precision
      const decimals = contract.decimals || 8;

      // Parse the input which may have decimals (e.g., "123.456")
      const parts = airdropAmount.split('.');
      const wholePart = parts[0] || '0';
      const fracPart = (parts[1] || '').slice(0, decimals).padEnd(decimals, '0');

      // Combine whole and fractional parts as BigInt in base units
      const baseUnits = BigInt(wholePart) * BigInt(10) ** BigInt(decimals) + BigInt(fracPart);

      console.log(`✈️ Airdropping ${airdropAmount} ${contract.symbol} (${baseUnits.toString()} base units each) to ${addresses.length} addresses`);

      const response = await fetch('/api/v1/contracts/airdrop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contract.address,
          recipients: addresses,
          amount_per_recipient: baseUnits.toString(), // v1.4.10: Send base units, not display units
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to airdrop tokens');
      }

      const result = await response.json();
      console.log('✅ Airdrop successful:', result);

      // v1.4.10: Add airdrop event to event history
      const newEvent: ContractEvent = {
        id: `airdrop-${Date.now()}`,
        type: 'airdrop',
        amount: airdropAmount, // Display units for UI (per recipient)
        recipients: addresses.length,
        timestamp: new Date(),
        txHash: result.data?.transaction_hash || `0x${Date.now().toString(16)}`
      };
      setContractEvents(prev => ({
        ...prev,
        [contract.address]: [newEvent, ...(prev[contract.address] || [])]
      }));

      alert(`Successfully airdropped ${airdropAmount} ${contract.symbol} to ${addresses.length} addresses!`);
      setAirdropAddresses('');
      setAirdropAmount('');
    } catch (error: any) {
      console.error('❌ Airdrop failed:', error);
      alert(`Airdrop failed: ${error.message}`);
    }
  };

  const handleTogglePause = async (contract: DeployedContract) => {
    try {
      const newPauseState = !contract.isPaused;
      console.log(`${newPauseState ? 'Pausing' : 'Resuming'} contract ${contract.address}`);

      const response = await fetch('/api/v1/contracts/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contract.address,
          paused: newPauseState,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to update pause state');
      }

      const result = await response.json();
      console.log('✅ Pause state updated:', result);

      // Update local state
      setDeployedContracts(prev =>
        prev.map(c =>
          c.address === contract.address
            ? { ...c, isPaused: newPauseState }
            : c
        )
      );

      alert(`Contract ${newPauseState ? 'paused' : 'resumed'} successfully!`);
    } catch (error: any) {
      console.error('❌ Failed to update pause state:', error);
      alert(`Failed to ${contract.isPaused ? 'resume' : 'pause'} contract: ${error.message}`);
    }
  };

  const handleUpdateReflection = async (contract: DeployedContract) => {
    if (!reflectionRate || parseFloat(reflectionRate) < 0 || parseFloat(reflectionRate) > 10) {
      alert('Please enter a valid reflection rate between 0% and 10%');
      return;
    }

    try {
      console.log(`Setting reflection rate to ${reflectionRate}% for ${contract.symbol}`);

      const response = await fetch('/api/v1/contracts/reflection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_address: contract.address,
          rate: reflectionRate,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to update reflection rate');
      }

      const result = await response.json();
      console.log('✅ Reflection rate updated:', result);
      alert(`Reflection rate updated to ${reflectionRate}% for ${contract.symbol}!`);
    } catch (error: any) {
      console.error('❌ Failed to update reflection rate:', error);
      alert(`Failed to update reflection rate: ${error.message}`);
    }
  };

  // v4.0.3: Generic RWA contract action handler with inline feedback
  const handleContractAction = async (contract: DeployedContract, action: string, extraData?: Record<string, string>) => {
    setRwaStatus(contract.address, action, { loading: true, message: 'Processing...', success: false });
    try {
      console.log(`🏗️ RWA action "${action}" on contract ${contract.address}`, extraData);
      const response = await fetch(`/api/v1/contracts/${contract.address}/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...extraData }),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `Action "${action}" failed`);
      }
      const result = await response.json();
      console.log(`✅ RWA action "${action}" succeeded:`, result);

      // Add event to contract history
      const newEvent: ContractEvent = {
        id: `${action}-${Date.now()}`,
        type: 'transfer',
        amount: extraData?.amount || '0',
        timestamp: new Date(),
        txHash: result.data?.transaction_hash || `0x${Date.now().toString(16)}`
      };
      setContractEvents(prev => ({
        ...prev,
        [contract.address]: [newEvent, ...(prev[contract.address] || [])]
      }));

      const readableAction = action.replace(/_/g, ' ');
      setRwaStatus(contract.address, action, { loading: false, message: `${readableAction} completed`, success: true });

      // Refresh contract data to show updated params (e.g. new appraisal value, verification status)
      if (result.data?.updated_params) {
        setDeployedContracts(prev => prev.map(c =>
          c.address === contract.address
            ? { ...c, deploymentParams: { ...c.deploymentParams, ...result.data.updated_params } }
            : c
        ));
      }
    } catch (error: any) {
      console.error(`❌ RWA action "${action}" failed:`, error);
      setRwaStatus(contract.address, action, { loading: false, message: error.message, success: false });
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="p-3 rainbow-box rounded-xl">
          <Cpu className="w-8 h-8 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-quantum-cyan to-quantum-purple bg-clip-text text-transparent">
            qVM
          </h1>
          <p className="text-gray-400">
            Deploy quantum-safe smart contracts in seconds
          </p>
        </div>
      </div>

      {/* ═══ Top-Level Navigation Tabs ═══ */}
      <div className="flex gap-2 border-b border-quantum-purple/10 pb-3">
        {[
          { id: 'deploy' as const, label: 'Deploy & Templates', icon: Cpu },
          { id: 'contracts' as const, label: 'My Contracts', icon: FileCode, count: deployedContracts.length },
          { id: 'portfolio' as const, label: 'RWA Portfolio', icon: Wallet },
        ].map(tab => (
          <motion.button
            key={tab.id}
            onClick={() => { setVmView(tab.id); if (tab.id === 'deploy') setStep('select'); }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all ${
              vmView === tab.id
                ? 'bg-gradient-to-r from-quantum-purple/80 to-quantum-cyan/80 text-white shadow-lg'
                : 'bg-quantum-dark/40 text-gray-400 hover:text-white hover:bg-quantum-dark/60'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
            {tab.count !== undefined && tab.count > 0 && (
              <span className="bg-quantum-purple/30 text-xs px-1.5 py-0.5 rounded-full">{tab.count}</span>
            )}
          </motion.button>
        ))}
      </div>

      {/* ═══ RWA Portfolio View ═══ */}
      {vmView === 'portfolio' && (
        <RwaPortfolioTab
          contracts={deployedContracts}
          walletAddress={localStorage.getItem('walletAddress') || ''}
        />
      )}

      {/* ═══ My Contracts View (standalone) ═══ */}
      {vmView === 'contracts' && (loadingContracts || deployedContracts.length > 0) && (() => {
        // This renders the deployed contracts section directly
        // We reuse the same section that's currently inside step === 'select'
        return null; // Will be rendered below alongside select step
      })()}

      <AnimatePresence mode="wait">
        {/* Step 1: Contract Selection */}
        {(vmView === 'deploy' || vmView === 'contracts') && step === 'select' && (
          <motion.div
            key="select"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Category Tabs + Template Grid - only show in deploy view */}
            {vmView === 'deploy' && (<>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {categories.map((category) => (
                <motion.button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all whitespace-nowrap ${
                    selectedCategory === category.id
                      ? 'bg-gradient-to-r from-quantum-purple to-quantum-cyan text-white'
                      : 'bg-quantum-indigo/30 text-gray-400 hover:text-white hover:bg-quantum-purple/20'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <category.icon className="w-5 h-5" />
                  {category.name}
                </motion.button>
              ))}
            </div>

            {/* Contract Templates Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTemplates.map((template) => (
                <motion.div
                  key={template.id}
                  onClick={() => handleSelectTemplate(template)}
                  className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6 cursor-pointer hover:border-quantum-cyan/50 transition-all group"
                  whileHover={{ scale: 1.02, y: -4 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-3 bg-quantum-purple/20 rounded-lg group-hover:bg-quantum-cyan/20 transition-colors">
                      <template.icon className="w-6 h-6 text-quantum-purple group-hover:text-quantum-cyan transition-colors" />
                    </div>
                    <span className="text-xs text-quantum-green font-mono">{template.gasEstimate}</span>
                  </div>

                  <h3 className="text-lg font-bold text-white mb-2">{template.name}</h3>
                  <p className="text-sm text-gray-400 mb-4">{template.description}</p>

                  <div className="flex flex-wrap gap-2">
                    {template.features.slice(0, 3).map((feature) => (
                      <span key={feature} className="text-xs bg-quantum-purple/20 text-quantum-purple px-2 py-1 rounded">
                        {feature}
                      </span>
                    ))}
                    {template.features.length > 3 && (
                      <span className="text-xs text-gray-500">+{template.features.length - 3} more</span>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
            </>)}
          </motion.div>
        )}

        {/* Step 2: Basic Information */}
        {step === 'basics' && selectedTemplate && (
          <motion.div
            key="basics"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-6">
                <selectedTemplate.icon className="w-6 h-6 text-quantum-cyan" />
                <h2 className="text-2xl font-bold text-white">{selectedTemplate.name}</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Contract Name
                  </label>
                  <input
                    type="text"
                    value={contractName}
                    onChange={(e) => setContractName(e.target.value)}
                    placeholder="My Token"
                    className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none"
                  />
                </div>

                {/* v4.0.1: Description field for all contracts */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Description <span className="text-gray-500 text-xs">(Optional — visible on social profile)</span>
                  </label>
                  <textarea
                    value={contractDescription}
                    onChange={(e) => setContractDescription(e.target.value)}
                    placeholder={isRwaTemplate
                      ? "Describe your real-world asset token — what it represents, investment thesis, target market..."
                      : "Describe your token project — purpose, use case, community..."
                    }
                    rows={3}
                    maxLength={500}
                    className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none resize-none text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">{contractDescription.length}/500 characters. Can be edited later in the Social tab.</p>
                </div>

                {selectedTemplate.id.includes('token') && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Token Symbol
                      </label>
                      <input
                        type="text"
                        value={tokenSymbol}
                        onChange={(e) => setTokenSymbol(e.target.value.toUpperCase())}
                        placeholder="MTK"
                        maxLength={5}
                        className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none"
                      />
                    </div>

                    {/* v3.2.18-beta: Initial Supply with Slider + Max Button */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Initial Supply
                      </label>
                      <div className="flex gap-2 mb-3">
                        <input
                          type="text"
                          value={initialSupply}
                          onChange={(e) => {
                            const val = e.target.value.replace(/[^0-9]/g, '');
                            setInitialSupply(val);
                          }}
                          placeholder="1000000"
                          className="flex-1 bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none font-mono"
                        />
                        <motion.button
                          type="button"
                          onClick={() => {
                            // u128 max with selected decimals: 3.4e38 / 10^decimals
                            // For safety, use 10^(38-decimals) as practical max
                            const maxExponent = Math.max(1, 38 - tokenDecimals);
                            const maxSupply = '1' + '0'.repeat(maxExponent);
                            setInitialSupply(maxSupply);
                          }}
                          className="px-4 py-2 bg-gradient-to-r from-quantum-purple/50 to-quantum-cyan/50 hover:from-quantum-purple/70 hover:to-quantum-cyan/70 border border-quantum-cyan/30 rounded-lg text-quantum-cyan text-sm font-bold transition-all"
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          MAX
                        </motion.button>
                      </div>
                      {/* Gradient Slider for Initial Supply - logarithmic scale for u128 range */}
                      <div className="relative">
                        <input
                          type="range"
                          min="3"
                          max="36"
                          step="1"
                          value={Math.max(3, Math.min(36, Math.log10(Number(initialSupply) || 1000)))}
                          onChange={(e) => {
                            const exp = Number(e.target.value);
                            // v3.6.18: Use BigInt to preserve precision for large exponents (>15)
                            // Math.pow(10, 30) loses precision, but BigInt(10) ** BigInt(30) is exact
                            setInitialSupply((BigInt(10) ** BigInt(exp)).toString());
                          }}
                          className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                          style={{
                            background: `linear-gradient(to right, #8B5CF6, #06B6D4, #10B981, #F59E0B)`,
                          }}
                        />
                        <style>{`
                          input[type="range"]::-webkit-slider-thumb {
                            appearance: none;
                            width: 20px;
                            height: 20px;
                            background: linear-gradient(135deg, #8B5CF6, #06B6D4);
                            border-radius: 50%;
                            cursor: pointer;
                            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5), 0 0 20px rgba(6, 182, 212, 0.3);
                            border: 2px solid rgba(255, 255, 255, 0.3);
                          }
                          input[type="range"]::-moz-range-thumb {
                            width: 20px;
                            height: 20px;
                            background: linear-gradient(135deg, #8B5CF6, #06B6D4);
                            border-radius: 50%;
                            cursor: pointer;
                            box-shadow: 0 0 10px rgba(139, 92, 246, 0.5), 0 0 20px rgba(6, 182, 212, 0.3);
                            border: 2px solid rgba(255, 255, 255, 0.3);
                          }
                        `}</style>
                        {/* Labels positioned to match logarithmic scale (3-36) */}
                        <div className="relative h-5 mt-1">
                          <span className="absolute text-xs text-gray-500" style={{ left: '0%' }}>1K</span>
                          <span className="absolute text-xs text-gray-500" style={{ left: '18%', transform: 'translateX(-50%)' }}>1B</span>
                          <span className="absolute text-xs text-gray-500" style={{ left: '27%', transform: 'translateX(-50%)' }}>1T</span>
                          <span className="absolute text-xs text-gray-500" style={{ left: '64%', transform: 'translateX(-50%)' }}>10^24</span>
                          <span className="absolute text-xs text-gray-500" style={{ right: '0%' }}>10^36</span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Display value: <span className="text-quantum-cyan font-mono">{(() => {
                          try {
                            const val = BigInt(initialSupply || '0');
                            // Format large numbers with scientific notation if too big
                            if (val > BigInt('1000000000000000')) {
                              const str = val.toString();
                              return `${str[0]}.${str.slice(1,4)}... × 10^${str.length - 1}`;
                            }
                            return val.toLocaleString();
                          } catch { return initialSupply; }
                        })()}</span> tokens
                        <span className="ml-2 text-quantum-purple">(u128 supports up to ~10^{38 - tokenDecimals} with {tokenDecimals} decimals)</span>
                      </p>
                    </div>

                    {/* v3.2.18-beta: Token Decimals Slider - constrained by supply for u128 */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Token Decimals
                      </label>
                      {(() => {
                        // u128 max ≈ 3.4 × 10^38, so max_decimals = 38 - log10(supply)
                        const supplyDigits = (initialSupply || '1').length;
                        const maxDecimalsForSupply = Math.max(0, Math.min(24, 38 - supplyDigits));
                        const effectiveDecimals = Math.min(tokenDecimals, maxDecimalsForSupply);

                        // Auto-adjust if current decimals exceeds max
                        if (tokenDecimals > maxDecimalsForSupply) {
                          setTimeout(() => setTokenDecimals(maxDecimalsForSupply), 0);
                        }

                        return (
                          <>
                            <div className="flex items-center gap-4 mb-2">
                              <div className="relative flex-1">
                                <input
                                  type="range"
                                  min="0"
                                  max={maxDecimalsForSupply}
                                  step="1"
                                  value={effectiveDecimals}
                                  onChange={(e) => setTokenDecimals(Number(e.target.value))}
                                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                                  style={{
                                    background: `linear-gradient(to right, #F59E0B ${(effectiveDecimals / Math.max(1, maxDecimalsForSupply)) * 100}%, #374151 ${(effectiveDecimals / Math.max(1, maxDecimalsForSupply)) * 100}%)`,
                                  }}
                                />
                              </div>
                              <div className="w-16 text-center">
                                <span className="text-xl font-bold text-quantum-cyan">{effectiveDecimals}</span>
                              </div>
                            </div>
                            {/* Labels positioned to match linear scale (0 to maxDecimalsForSupply) */}
                            <div className="relative h-5">
                              <span className="absolute text-xs text-gray-500" style={{ left: '0%' }}>0</span>
                              {maxDecimalsForSupply >= 8 && (
                                <span className="absolute text-xs text-gray-500" style={{ left: `${(8 / Math.max(1, maxDecimalsForSupply)) * 100}%`, transform: 'translateX(-50%)' }}>8</span>
                              )}
                              {maxDecimalsForSupply >= 18 && (
                                <span className="absolute text-xs text-gray-500" style={{ left: `${(18 / Math.max(1, maxDecimalsForSupply)) * 100}%`, transform: 'translateX(-50%)' }}>18</span>
                              )}
                              <span className={`absolute text-xs ${maxDecimalsForSupply < 24 ? 'text-red-400' : 'text-gray-500'}`} style={{ right: '0%' }}>{maxDecimalsForSupply}</span>
                            </div>
                            {maxDecimalsForSupply < 24 && (
                              <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded-lg">
                                <p className="text-xs text-red-400">
                                  u128 limit: With {supplyDigits}-digit supply, max decimals is {maxDecimalsForSupply}
                                </p>
                              </div>
                            )}
                            <div className="mt-2 p-3 bg-quantum-dark/30 rounded-lg border border-quantum-purple/20">
                              <p className="text-xs text-gray-400">
                                <span className="text-quantum-cyan">{effectiveDecimals}</span> decimals = 1 token = <span className="font-mono text-quantum-purple">10^{effectiveDecimals}</span> smallest units
                              </p>
                              <p className="text-xs text-gray-500 mt-1">
                                Smallest unit: <span className="font-mono text-quantum-cyan">0.{'0'.repeat(Math.max(0, effectiveDecimals - 1))}1</span>
                              </p>
                              <p className="text-xs text-gray-500 mt-1">
                                {effectiveDecimals === 8 ? '(Standard - recommended for most tokens)' :
                                 effectiveDecimals === 18 ? '(Ethereum compatible)' :
                                 effectiveDecimals === 24 ? '(QUG precision - maximum granularity)' :
                                 effectiveDecimals === 0 ? '(NFT-like - whole tokens only)' : ''}
                              </p>
                            </div>
                          </>
                        );
                      })()}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Token Logo (Optional)
                      </label>
                      <div className="flex gap-3">
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/png"
                          onChange={handleLogoUpload}
                          className="hidden"
                        />
                        <motion.button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="flex-1 bg-quantum-dark/50 border border-quantum-cyan/20 hover:border-quantum-cyan/50 rounded-lg px-4 py-3 text-gray-300 transition-colors flex items-center justify-center gap-2"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <Upload className="w-4 h-4" />
                          {logoFile ? logoFile.name : 'Upload PNG Logo'}
                        </motion.button>
                        {logoPreview && (
                          <div className="w-16 h-16 bg-quantum-dark/50 border border-quantum-cyan/30 rounded-lg flex items-center justify-center overflow-hidden">
                            <img src={logoPreview} alt="Logo preview" className="w-full h-full object-contain" />
                          </div>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        PNG format, max 500KB. Stored on libp2p/IPFS.
                      </p>
                    </div>
                  </>
                )}
              </div>

              <div className="flex gap-3 mt-6">
                <motion.button
                  onClick={() => setStep('select')}
                  className="flex-1 bg-quantum-dark/50 hover:bg-quantum-dark/70 text-white py-3 px-4 rounded-xl transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Back
                </motion.button>
                <motion.button
                  onClick={() => (selectedTemplate.id === 'advanced-token' || isRwaTemplate) ? setStep('features') : setStep('review')}
                  disabled={!contractName || (selectedTemplate.id.includes('token') && (!tokenSymbol || !initialSupply))}
                  className="flex-1 bg-gradient-to-r from-quantum-purple to-quantum-cyan hover:from-quantum-purple/80 hover:to-quantum-cyan/80 text-white py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Next
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 3: Features Selection (for Advanced Token) */}
        {step === 'features' && (
          <motion.div
            key="features"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
              {isRwaTemplate ? (
                <>
                  <div className="flex items-center gap-3 mb-2">
                    {selectedTemplate && <selectedTemplate.icon className="w-6 h-6 text-quantum-cyan" />}
                    <h2 className="text-2xl font-bold text-white">Configure {selectedTemplate?.name}</h2>
                  </div>
                  <p className="text-gray-400 mb-6">Set the specific parameters for your RWA contract</p>

                  <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
                    {/* Group fields by section */}
                    {(() => {
                      const sections = new Map<string, RwaField[]>();
                      currentRwaFields.forEach(field => {
                        const sec = field.section || 'General';
                        if (!sections.has(sec)) sections.set(sec, []);
                        sections.get(sec)!.push(field);
                      });
                      return Array.from(sections.entries()).map(([sectionName, fields]) => (
                        <div key={sectionName}>
                          <h3 className="text-lg font-bold text-quantum-cyan mb-3">{sectionName}</h3>
                          <div className="space-y-3">
                            {fields.map(field => {
                              if (field.type === 'checkbox') {
                                return (
                                  <label key={field.key} className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                                    <input
                                      type="checkbox"
                                      checked={!!rwaParams[field.key]}
                                      onChange={() => updateRwaParam(field.key, !rwaParams[field.key])}
                                      className="w-5 h-5 rounded border-quantum-cyan/30 text-quantum-cyan focus:ring-quantum-cyan"
                                    />
                                    <div>
                                      <div className="font-medium text-white">{field.label}</div>
                                      {field.description && <div className="text-sm text-gray-400">{field.description}</div>}
                                    </div>
                                  </label>
                                );
                              } else if (field.type === 'select') {
                                return (
                                  <div key={field.key}>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">{field.label}</label>
                                    <select
                                      value={(rwaParams[field.key] as string) || ''}
                                      onChange={(e) => updateRwaParam(field.key, e.target.value)}
                                      className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white focus:border-quantum-cyan/50 focus:outline-none appearance-none"
                                    >
                                      <option value="" className="bg-gray-900">Select {field.label}...</option>
                                      {field.options?.map(opt => (
                                        <option key={opt.value} value={opt.value} className="bg-gray-900">{opt.label}</option>
                                      ))}
                                    </select>
                                  </div>
                                );
                              } else {
                                return (
                                  <div key={field.key}>
                                    <label className="block text-sm font-medium text-gray-300 mb-2">{field.label}</label>
                                    <input
                                      type={field.type === 'number' ? 'text' : 'text'}
                                      value={(rwaParams[field.key] as string) || ''}
                                      onChange={(e) => {
                                        const val = field.type === 'number' ? e.target.value.replace(/[^0-9.]/g, '') : e.target.value;
                                        updateRwaParam(field.key, val);
                                      }}
                                      placeholder={field.placeholder}
                                      className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none"
                                    />
                                  </div>
                                );
                              }
                            })}
                          </div>
                        </div>
                      ));
                    })()}
                  </div>
                </>
              ) : (
                <>
              <h2 className="text-2xl font-bold text-white mb-2">Select Features</h2>
              <p className="text-gray-400 mb-6">Choose the capabilities you want for your token</p>

              <div className="space-y-6">
                {/* Supply Management */}
                <div>
                  <h3 className="text-lg font-bold text-quantum-cyan mb-3">Supply Management</h3>
                  <div className="space-y-3">
                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.mintable}
                        onChange={() => toggleFeature('mintable')}
                        className="w-5 h-5 rounded border-quantum-cyan/30 text-quantum-cyan focus:ring-quantum-cyan"
                      />
                      <div>
                        <div className="font-medium text-white">Mintable</div>
                        <div className="text-sm text-gray-400">Enable minting new tokens</div>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.burnable}
                        onChange={() => toggleFeature('burnable')}
                        className="w-5 h-5 rounded border-quantum-cyan/30 text-quantum-cyan focus:ring-quantum-cyan"
                      />
                      <div>
                        <div className="font-medium text-white">Burnable</div>
                        <div className="text-sm text-gray-400">Enable burning/destroying tokens</div>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.airdrop}
                        onChange={() => toggleFeature('airdrop')}
                        className="w-5 h-5 rounded border-quantum-cyan/30 text-quantum-cyan focus:ring-quantum-cyan"
                      />
                      <div>
                        <div className="font-medium text-white">Airdrop</div>
                        <div className="text-sm text-gray-400">Enable bulk distribution to multiple addresses</div>
                      </div>
                    </label>
                  </div>
                </div>

                {/* Holder Rewards */}
                <div>
                  <h3 className="text-lg font-bold text-quantum-purple mb-3">Holder Rewards</h3>
                  <div className="space-y-3">
                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.reflection}
                        onChange={() => toggleFeature('reflection')}
                        className="w-5 h-5 rounded border-quantum-purple/30 text-quantum-purple focus:ring-quantum-purple"
                      />
                      <div>
                        <div className="font-medium text-white">Reflection</div>
                        <div className="text-sm text-gray-400">Redistribute fees to holders</div>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.staking}
                        onChange={() => toggleFeature('staking')}
                        className="w-5 h-5 rounded border-quantum-purple/30 text-quantum-purple focus:ring-quantum-purple"
                      />
                      <div>
                        <div className="font-medium text-white">Staking</div>
                        <div className="text-sm text-gray-400">Enable staking functionality</div>
                      </div>
                    </label>
                  </div>
                </div>

                {/* Governance & Control */}
                <div>
                  <h3 className="text-lg font-bold text-quantum-green mb-3">Governance & Control</h3>
                  <div className="space-y-3">
                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.governance}
                        onChange={() => toggleFeature('governance')}
                        className="w-5 h-5 rounded border-quantum-green/30 text-quantum-green focus:ring-quantum-green"
                      />
                      <div>
                        <div className="font-medium text-white">Governance</div>
                        <div className="text-sm text-gray-400">Enable voting rights for holders</div>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.pausable}
                        onChange={() => toggleFeature('pausable')}
                        className="w-5 h-5 rounded border-quantum-green/30 text-quantum-green focus:ring-quantum-green"
                      />
                      <div>
                        <div className="font-medium text-white">Pausable</div>
                        <div className="text-sm text-gray-400">Allow pausing contract in emergencies</div>
                      </div>
                    </label>

                    <label className="flex items-center gap-3 p-4 bg-quantum-dark/50 rounded-lg cursor-pointer hover:bg-quantum-dark/70 transition-colors">
                      <input
                        type="checkbox"
                        checked={features.upgradeable}
                        onChange={() => toggleFeature('upgradeable')}
                        className="w-5 h-5 rounded border-quantum-green/30 text-quantum-green focus:ring-quantum-green"
                      />
                      <div>
                        <div className="font-medium text-white">Upgradeable</div>
                        <div className="text-sm text-gray-400">Allow upgrading contract logic</div>
                      </div>
                    </label>
                  </div>
                </div>
              </div>
                </>
              )}

              <div className="flex gap-3 mt-6">
                <motion.button
                  onClick={() => setStep('basics')}
                  className="flex-1 bg-quantum-dark/50 hover:bg-quantum-dark/70 text-white py-3 px-4 rounded-xl transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Back
                </motion.button>
                <motion.button
                  onClick={() => setStep('review')}
                  className="flex-1 bg-gradient-to-r from-quantum-purple to-quantum-cyan hover:from-quantum-purple/80 hover:to-quantum-cyan/80 text-white py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Review
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 4: Review & Deploy */}
        {step === 'review' && selectedTemplate && (
          <motion.div
            key="review"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-6">Review & Deploy</h2>

              <div className="space-y-4 mb-6">
                <div className="bg-quantum-dark/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Contract Type</div>
                  <div className="text-lg font-bold text-white">{selectedTemplate.name}</div>
                </div>

                <div className="bg-quantum-dark/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Contract Name</div>
                  <div className="text-lg font-bold text-white">{contractName}</div>
                </div>

                {tokenSymbol && (
                  <div className="bg-quantum-dark/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Token Symbol</div>
                    <div className="text-lg font-bold text-white">{tokenSymbol}</div>
                  </div>
                )}

                {initialSupply && (
                  <div className="bg-quantum-dark/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Initial Supply</div>
                    <div className="text-lg font-bold text-white">{(() => {
                      // v3.2.14-beta: Use BigInt for large number display to prevent precision loss
                      try {
                        return BigInt(initialSupply).toLocaleString();
                      } catch {
                        return initialSupply;
                      }
                    })()} {tokenSymbol}</div>
                    {/* v3.2.18-beta: Show decimals context with the supply */}
                    <div className="text-xs text-gray-500 mt-1">
                      {tokenDecimals} decimals • Smallest unit: 0.{'0'.repeat(Math.max(0, tokenDecimals - 1))}1 {tokenSymbol}
                    </div>
                  </div>
                )}

                {selectedTemplate.id === 'advanced-token' && (
                  <div className="bg-quantum-dark/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">Enabled Features</div>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(features)
                        .filter(([_, enabled]) => enabled)
                        .map(([feature]) => (
                          <span key={feature} className="bg-quantum-cyan/20 text-quantum-cyan px-3 py-1 rounded-full text-sm capitalize">
                            {feature}
                          </span>
                        ))}
                    </div>
                  </div>
                )}

                {/* Description in review */}
                {contractDescription && (
                  <div className="bg-quantum-dark/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">Description</div>
                    <p className="text-white text-sm">{contractDescription}</p>
                  </div>
                )}

                {/* RWA Parameters Summary */}
                {isRwaTemplate && Object.keys(rwaParams).length > 0 && (
                  <div className="bg-quantum-dark/50 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">RWA Configuration</div>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(rwaParams)
                        .filter(([_, val]) => val !== '' && val !== false)
                        .map(([key, val]) => {
                          const field = currentRwaFields.find(f => f.key === key);
                          const displayVal = typeof val === 'boolean' ? 'Yes' :
                            field?.options?.find(o => o.value === val)?.label || String(val);
                          return (
                            <div key={key} className="text-sm">
                              <span className="text-gray-500">{field?.label || key}: </span>
                              <span className="text-quantum-cyan">{displayVal}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                <div className="bg-gradient-to-r from-quantum-yellow/10 to-quantum-orange/10 border border-quantum-yellow/30 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-300">Estimated Gas Cost</div>
                    <div className="text-lg font-bold text-quantum-yellow">{selectedTemplate.gasEstimate}</div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3">
                <motion.button
                  onClick={() => setStep((selectedTemplate.id === 'advanced-token' || isRwaTemplate) ? 'features' : 'basics')}
                  className="flex-1 bg-quantum-dark/50 hover:bg-quantum-dark/70 text-white py-3 px-4 rounded-xl transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Back
                </motion.button>
                <motion.button
                  onClick={handleDeploy}
                  className="flex-1 bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 font-bold"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Code className="w-5 h-5" />
                  Deploy Contract
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 5: Deploying */}
        {step === 'deploying' && (
          <motion.div
            key="deploying"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="flex items-center justify-center py-20"
          >
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-12 text-center">
              <motion.div
                className="w-20 h-20 mx-auto mb-6 rainbow-box rounded-full flex items-center justify-center"
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <Cpu className="w-10 h-10 text-white" />
              </motion.div>
              <h2 className="text-2xl font-bold text-white mb-2">Deploying Contract...</h2>
              <p className="text-gray-400">Quantum consensus in progress</p>
            </div>
          </motion.div>
        )}

        {/* Step 6: Success */}
        {step === 'success' && selectedTemplate && (
          <motion.div
            key="success"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="space-y-6"
          >
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-green/30 rounded-xl p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", bounce: 0.5 }}
                className="w-20 h-20 mx-auto mb-6 bg-quantum-green/20 rounded-full flex items-center justify-center"
              >
                <CheckCircle className="w-12 h-12 text-quantum-green" />
              </motion.div>

              <h2 className="text-3xl font-bold text-white mb-2">Contract Deployed!</h2>
              <p className="text-gray-400 mb-6">Your smart contract is now live on Q-NarwhalKnight</p>

              <div className="bg-quantum-dark/50 rounded-lg p-4 mb-6">
                <div className="text-sm text-gray-400 mb-2">Contract Address</div>
                <div className="font-mono text-quantum-cyan break-all">
                  {lastDeployedAddress || 'Deploying...'}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-quantum-dark/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Type</div>
                  <div className="font-bold text-white">{selectedTemplate.name}</div>
                </div>
                <div className="bg-quantum-dark/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Name</div>
                  <div className="font-bold text-white">{contractName}</div>
                </div>
                <div className="bg-quantum-dark/50 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Symbol</div>
                  <div className="font-bold text-white">{tokenSymbol || 'N/A'}</div>
                </div>
              </div>

              <div className="flex gap-3">
                <motion.button
                  onClick={() => {
                    setStep('select');
                    setContractName('');
                    setTokenSymbol('');
                    setInitialSupply('1000000');
                    setSelectedTemplate(null);
                  }}
                  className="flex-1 bg-quantum-purple/30 hover:bg-quantum-purple/40 text-white py-3 px-4 rounded-xl transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Deploy Another
                </motion.button>
                <motion.button
                  onClick={() => setStep('select')}
                  className="flex-1 bg-gradient-to-r from-quantum-cyan to-quantum-blue hover:from-quantum-cyan/80 hover:to-quantum-blue/80 text-white py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Settings className="w-4 h-4" />
                  Manage Contract
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* My Deployed Contracts Section - visible in deploy (select step) and contracts views */}
      {(vmView === 'contracts' || (vmView === 'deploy' && step === 'select')) && (loadingContracts || deployedContracts.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          <h2 className="text-2xl font-bold text-white">My Deployed Contracts</h2>

          {/* Search, Sort & Filter Toolbar */}
          {deployedContracts.length > 0 && (
            <div className="bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/20 rounded-xl p-4 space-y-3">
              {/* Search Bar */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <input
                  type="text"
                  value={contractSearch}
                  onChange={(e) => setContractSearch(e.target.value)}
                  placeholder="Search contracts by name, symbol, type, or address..."
                  className="w-full bg-quantum-dark/60 border border-quantum-purple/20 rounded-lg pl-10 pr-10 py-2.5 text-white text-sm placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none transition-colors"
                />
                {contractSearch && (
                  <button onClick={() => setContractSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white transition-colors">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Filter & Sort Row */}
              <div className="flex flex-wrap gap-2 items-center">
                {/* Type Filter */}
                <div className="flex items-center gap-1.5">
                  <Filter className="w-3.5 h-3.5 text-gray-500" />
                  <select
                    value={contractTypeFilter}
                    onChange={(e) => setContractTypeFilter(e.target.value)}
                    className="bg-quantum-dark/60 border border-quantum-purple/20 rounded-lg px-3 py-1.5 text-white text-xs focus:border-quantum-cyan/50 focus:outline-none appearance-none cursor-pointer"
                  >
                    <option value="all">All Types</option>
                    <option value="token">Tokens</option>
                    <option value="rwa">RWA</option>
                    <option value="governance">Governance</option>
                    <option value="defi">DeFi</option>
                  </select>
                </div>

                {/* Sort */}
                <div className="flex items-center gap-1.5">
                  <ArrowUpDown className="w-3.5 h-3.5 text-gray-500" />
                  <select
                    value={contractSort}
                    onChange={(e) => setContractSort(e.target.value as 'name' | 'date' | 'type' | 'symbol')}
                    className="bg-quantum-dark/60 border border-quantum-purple/20 rounded-lg px-3 py-1.5 text-white text-xs focus:border-quantum-cyan/50 focus:outline-none appearance-none cursor-pointer"
                  >
                    <option value="date">Sort by Date</option>
                    <option value="name">Sort by Name</option>
                    <option value="symbol">Sort by Symbol</option>
                    <option value="type">Sort by Type</option>
                  </select>
                  <button
                    onClick={() => setContractSortDir(d => d === 'asc' ? 'desc' : 'asc')}
                    className="bg-quantum-dark/60 border border-quantum-purple/20 rounded-lg px-2 py-1.5 text-gray-400 hover:text-white text-xs transition-colors"
                    title={contractSortDir === 'asc' ? 'Ascending' : 'Descending'}
                  >
                    {contractSortDir === 'asc' ? '↑' : '↓'}
                  </button>
                </div>

                {/* Results count */}
                <span className="text-xs text-gray-500 ml-auto">
                  {filteredDeployedContracts.length} of {deployedContracts.length} contracts
                </span>
              </div>
            </div>
          )}

          {loadingContracts && (
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6 text-center">
              <p className="text-gray-400">Loading deployed contracts from blockchain...</p>
            </div>
          )}

          {/* No results message */}
          {!loadingContracts && deployedContracts.length > 0 && filteredDeployedContracts.length === 0 && (
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6 text-center">
              <Search className="w-8 h-8 text-gray-500 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">No contracts match your search</p>
              <button onClick={() => { setContractSearch(''); setContractTypeFilter('all'); }} className="text-quantum-cyan text-xs mt-2 hover:underline">
                Clear filters
              </button>
            </div>
          )}

          <div className="space-y-4">
            {filteredDeployedContracts.map((contract) => (
              <motion.div
                key={contract.address}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6"
              >
                {/* Contract Header */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-white mb-1">{contract.name}</h3>
                    <p className="text-sm text-gray-400">{contract.type}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-quantum-cyan">{contract.symbol}</span>
                    {contract.isPaused && (
                      <span className="bg-quantum-yellow/20 text-quantum-yellow text-xs px-2 py-1 rounded">
                        PAUSED
                      </span>
                    )}
                  </div>
                </div>

                {/* Contract Address */}
                <div className="bg-quantum-dark/50 rounded-lg p-3 mb-4">
                  <div className="text-xs text-gray-400 mb-1">Contract Address</div>
                  <div className="font-mono text-sm text-quantum-green break-all">{contract.address}</div>
                </div>

                {/* Token Balance */}
                {contract.abaBalance !== undefined && (
                  <div className="bg-gradient-to-r from-quantum-cyan/10 to-quantum-purple/10 border border-quantum-cyan/30 rounded-lg p-3 mb-4">
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-gray-400">Your Balance</div>
                      <div className="text-lg font-bold text-quantum-cyan">{contract.abaBalance} {contract.symbol}</div>
                    </div>
                  </div>
                )}

                {/* Tab Navigation */}
                <div className="flex gap-2 mb-4 border-b border-quantum-purple/20 pb-3">
                  <motion.button
                    onClick={() => setActiveTab(contract.address, 'control')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      getActiveTab(contract.address) === 'control'
                        ? 'bg-quantum-purple/30 text-white border border-quantum-purple/50'
                        : 'bg-quantum-dark/30 text-gray-400 hover:text-white hover:bg-quantum-dark/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Settings className="w-4 h-4" />
                    Control
                  </motion.button>
                  <motion.button
                    onClick={() => setActiveTab(contract.address, 'events')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      getActiveTab(contract.address) === 'events'
                        ? 'bg-quantum-cyan/30 text-white border border-quantum-cyan/50'
                        : 'bg-quantum-dark/30 text-gray-400 hover:text-white hover:bg-quantum-dark/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <History className="w-4 h-4" />
                    Events
                  </motion.button>
                  <motion.button
                    onClick={() => setActiveTab(contract.address, 'stats')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      getActiveTab(contract.address) === 'stats'
                        ? 'bg-quantum-green/30 text-white border border-quantum-green/50'
                        : 'bg-quantum-dark/30 text-gray-400 hover:text-white hover:bg-quantum-dark/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <BarChart3 className="w-4 h-4" />
                    Stats
                  </motion.button>
                  <motion.button
                    onClick={() => setActiveTab(contract.address, 'social')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      getActiveTab(contract.address) === 'social'
                        ? 'bg-quantum-pink/30 text-white border border-quantum-pink/50'
                        : 'bg-quantum-dark/30 text-gray-400 hover:text-white hover:bg-quantum-dark/50'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Users className="w-4 h-4" />
                    Social
                  </motion.button>
                </div>

                {/* Tab Content */}
                <AnimatePresence mode="wait">
                  {/* Control Tab */}
                  {getActiveTab(contract.address) === 'control' && (
                    <motion.div
                      key="control"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Settings className="w-4 h-4 text-quantum-purple" />
                      <h4 className="font-bold text-white">Contract Controls</h4>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      {/* Mint Control */}
                      {contract.features.mintable && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Sparkles className="w-4 h-4 text-quantum-green" />
                            <span className="font-medium text-white">Mint Tokens</span>
                          </div>
                          <div className="flex gap-2">
                            <input
                              type="text"
                              inputMode="numeric"
                              pattern="[0-9]*"
                              value={mintAmount}
                              onChange={(e) => {
                                // Only allow digits
                                const value = e.target.value.replace(/[^0-9]/g, '');
                                setMintAmount(value);
                              }}
                              placeholder="Amount"
                              className="flex-1 bg-quantum-dark/70 border border-quantum-green/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-green/50 focus:outline-none"
                            />
                            <motion.button
                              onClick={() => handleMint(contract)}
                              disabled={!mintAmount}
                              className="bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Mint
                            </motion.button>
                          </div>
                        </div>
                      )}

                      {/* Burn Control */}
                      {contract.features.burnable && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Flame className="w-4 h-4 text-quantum-orange" />
                            <span className="font-medium text-white">Burn Tokens</span>
                          </div>
                          <div className="flex gap-2">
                            <input
                              type="text"
                              inputMode="numeric"
                              pattern="[0-9]*"
                              value={burnAmount}
                              onChange={(e) => {
                                const value = e.target.value.replace(/[^0-9]/g, '');
                                setBurnAmount(value);
                              }}
                              placeholder="Amount"
                              className="flex-1 bg-quantum-dark/70 border border-quantum-orange/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-orange/50 focus:outline-none"
                            />
                            <motion.button
                              onClick={() => handleBurn(contract)}
                              disabled={!burnAmount}
                              className="bg-gradient-to-r from-quantum-orange to-quantum-red hover:from-quantum-orange/80 hover:to-quantum-red/80 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Burn
                            </motion.button>
                          </div>
                        </div>
                      )}

                      {/* Airdrop Control */}
                      {contract.features.airdrop && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4 md:col-span-2">
                          <div className="flex items-center gap-2 mb-3">
                            <Send className="w-4 h-4 text-quantum-blue" />
                            <span className="font-medium text-white">Airdrop Tokens</span>
                          </div>
                          <div className="space-y-2">
                            <textarea
                              value={airdropAddresses}
                              onChange={(e) => setAirdropAddresses(e.target.value)}
                              placeholder="Enter addresses (one per line or comma-separated)&#10;qnk1abc...&#10;qnk1def..."
                              rows={3}
                              className="w-full bg-quantum-dark/70 border border-quantum-blue/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-blue/50 focus:outline-none resize-none"
                            />
                            <div className="flex gap-2">
                              <input
                                type="text"
                                inputMode="numeric"
                                pattern="[0-9]*"
                                value={airdropAmount}
                                onChange={(e) => {
                                  const value = e.target.value.replace(/[^0-9]/g, '');
                                  setAirdropAmount(value);
                                }}
                                placeholder="Amount per address"
                                className="flex-1 bg-quantum-dark/70 border border-quantum-blue/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-blue/50 focus:outline-none"
                              />
                              <motion.button
                                onClick={() => handleAirdrop(contract)}
                                disabled={!airdropAddresses || !airdropAmount}
                                className="bg-gradient-to-r from-quantum-blue to-quantum-cyan hover:from-quantum-blue/80 hover:to-quantum-cyan/80 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                              >
                                Airdrop
                              </motion.button>
                            </div>
                            <p className="text-xs text-gray-500">
                              Send tokens to multiple addresses at once
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Reflection Control */}
                      {contract.features.reflection && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Zap className="w-4 h-4 text-quantum-purple" />
                            <span className="font-medium text-white">Reflection Rate</span>
                          </div>
                          <div className="flex gap-2">
                            <input
                              type="number"
                              value={reflectionRate}
                              onChange={(e) => setReflectionRate(e.target.value)}
                              placeholder="Rate %"
                              min="0"
                              max="10"
                              step="0.1"
                              className="flex-1 bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-purple/50 focus:outline-none"
                            />
                            <motion.button
                              onClick={() => handleUpdateReflection(contract)}
                              className="bg-gradient-to-r from-quantum-purple to-quantum-pink hover:from-quantum-purple/80 hover:to-quantum-pink/80 text-white px-4 py-2 rounded-lg text-sm font-medium"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Update
                            </motion.button>
                          </div>
                          <p className="text-xs text-gray-500 mt-2">
                            Current: {reflectionRate}% redistributed to holders
                          </p>
                        </div>
                      )}

                      {/* Staking Control */}
                      {contract.features.staking && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Users className="w-4 h-4 text-quantum-cyan" />
                            <span className="font-medium text-white">Staking Pool</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Total Staked:</span>
                              <span className="text-quantum-cyan font-medium">0 {contract.symbol}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">APY:</span>
                              <span className="text-quantum-green font-medium">12.5%</span>
                            </div>
                            <motion.button
                              className="w-full bg-quantum-cyan/20 hover:bg-quantum-cyan/30 text-quantum-cyan px-3 py-2 rounded-lg text-sm font-medium transition-colors"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Configure Staking
                            </motion.button>
                          </div>
                        </div>
                      )}

                      {/* Pause Control */}
                      {contract.features.pausable && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            {contract.isPaused ? (
                              <PlayCircle className="w-4 h-4 text-quantum-green" />
                            ) : (
                              <PauseCircle className="w-4 h-4 text-quantum-yellow" />
                            )}
                            <span className="font-medium text-white">Emergency Controls</span>
                          </div>
                          <motion.button
                            onClick={() => handleTogglePause(contract)}
                            className={`w-full ${
                              contract.isPaused
                                ? 'bg-gradient-to-r from-quantum-green to-quantum-cyan hover:from-quantum-green/80 hover:to-quantum-cyan/80'
                                : 'bg-gradient-to-r from-quantum-yellow to-quantum-orange hover:from-quantum-yellow/80 hover:to-quantum-orange/80'
                            } text-white px-4 py-2 rounded-lg text-sm font-medium`}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            {contract.isPaused ? 'Resume Contract' : 'Pause Contract'}
                          </motion.button>
                          <p className="text-xs text-gray-500 mt-2">
                            {contract.isPaused
                              ? 'Contract is currently paused - no transfers allowed'
                              : 'Pause all contract operations in case of emergency'}
                          </p>
                        </div>
                      )}

                      {/* Governance */}
                      {contract.features.governance && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Vote className="w-4 h-4 text-quantum-purple" />
                            <span className="font-medium text-white">Governance</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Active Proposals:</span>
                              <span className="text-white font-medium">0</span>
                            </div>
                            <motion.button
                              className="w-full bg-quantum-purple/20 hover:bg-quantum-purple/30 text-quantum-purple px-3 py-2 rounded-lg text-sm font-medium transition-colors"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Create Proposal
                            </motion.button>
                          </div>
                        </div>
                      )}

                      {/* Upgradeable */}
                      {contract.features.upgradeable && (
                        <div className="bg-quantum-dark/50 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <RefreshCw className="w-4 h-4 text-quantum-cyan" />
                            <span className="font-medium text-white">Contract Upgrade</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-400">Version:</span>
                              <span className="text-white font-medium">1.0.0</span>
                            </div>
                            <motion.button
                              className="w-full bg-quantum-cyan/20 hover:bg-quantum-cyan/30 text-quantum-cyan px-3 py-2 rounded-lg text-sm font-medium transition-colors"
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              Upload New Version
                            </motion.button>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* ═══ RWA-Specific Controls ═══ */}
                    {(() => {
                      const ct = contract.type.toLowerCase();
                      const isRwa = ['realestate', 'equity', 'fixedincome', 'commodity', 'carboncredit',
                        'artcollectible', 'iprevenue', 'physicalgoods', 'rwatoken', 'real_estate',
                        'fixed_income', 'carbon_credit', 'art_collectible', 'ip_revenue', 'physical_goods'
                      ].some(t => ct.includes(t));
                      const isRealEstate = ct.includes('realestate') || ct.includes('real_estate');
                      const isEquity = ct.includes('equity');
                      const isFixedIncome = ct.includes('fixedincome') || ct.includes('fixed_income');
                      const isCommodity = ct.includes('commodity');
                      const isCarbonCredit = ct.includes('carboncredit') || ct.includes('carbon_credit');
                      const isArt = ct.includes('artcollectible') || ct.includes('art_collectible');
                      const isIP = ct.includes('iprevenue') || ct.includes('ip_revenue');
                      const isPhysical = ct.includes('physicalgoods') || ct.includes('physical_goods');
                      if (!isRwa) return null;
                      return (
                      <div className="mt-4 space-y-4">
                        <div className="flex items-center gap-2 mb-1">
                          <Building className="w-4 h-4 text-amber-400" />
                          <h4 className="font-bold text-amber-400 tracking-wide text-sm uppercase">RWA Asset Management</h4>
                        </div>

                        {/* v4.0.1: Description display + quick edit */}
                        {(() => {
                          const desc = socialProfiles[contract.address]?.description;
                          return (
                            <div className="bg-gradient-to-r from-amber-900/5 to-quantum-dark/30 rounded-lg p-4 border border-amber-500/10">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-amber-400/70 font-medium uppercase tracking-wider">Description</span>
                                <motion.button
                                  onClick={() => {
                                    setActiveTab(contract.address, 'social');
                                    setEditingSocial(contract.address);
                                    setSocialFormData(socialProfiles[contract.address] || {});
                                  }}
                                  className="text-xs text-quantum-cyan hover:text-white transition-colors flex items-center gap-1"
                                  whileHover={{ scale: 1.05 }}
                                >
                                  <FileCode className="w-3 h-3" />
                                  {desc ? 'Edit' : 'Add Description'}
                                </motion.button>
                              </div>
                              {desc ? (
                                <p className="text-sm text-gray-300 leading-relaxed">{desc}</p>
                              ) : (
                                <p className="text-sm text-gray-500 italic">No description set. Click "Add Description" or go to the Social tab to add one.</p>
                              )}
                            </div>
                          );
                        })()}

                        <div className="grid md:grid-cols-2 gap-4">
                          {/* KYC/Compliance Management - show for all RWA types */}
                          {(isRwa) && (
                            <div className="bg-gradient-to-b from-amber-900/10 to-quantum-dark/50 rounded-lg p-4 border border-amber-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Shield className="w-4 h-4 text-amber-400" />
                                <span className="font-medium text-white">Compliance Controls</span>
                                <RwaInfoTip text="Manage regulatory compliance for this RWA token. KYC, accreditation, and transfer restrictions are enforced on-chain during DEX swaps." />
                              </div>
                              <div className="space-y-2">
                                {contract.features.kyc_required && (
                                  <div className="flex items-center justify-between bg-quantum-dark/40 rounded p-2">
                                    <span className="text-sm text-gray-300">KYC Required<RwaInfoTip text="Know Your Customer verification requires all token holders to verify their identity. When enabled, the DEX blocks purchases from unverified wallets." /></span>
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs text-amber-400">Active</span>
                                      <motion.button
                                        onClick={() => handleContractAction(contract, 'toggle_kyc')}
                                        className="bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 px-2 py-1 rounded text-xs"
                                        whileTap={{ scale: 0.95 }}
                                      >
                                        Configure
                                      </motion.button>
                                    </div>
                                  </div>
                                )}
                                {contract.features.accredited_only && (
                                  <div className="flex items-center justify-between bg-quantum-dark/40 rounded p-2">
                                    <span className="text-sm text-gray-300">Accredited Only<RwaInfoTip text="Restricts token ownership to accredited investors meeting SEC income ($200K+/yr) or net worth ($1M+) requirements." /></span>
                                    <span className="text-xs text-amber-400">Enforced</span>
                                  </div>
                                )}
                                {contract.features.transfer_restrictions && (
                                  <div className="flex items-center justify-between bg-quantum-dark/40 rounded p-2">
                                    <span className="text-sm text-gray-300">Transfer Restrictions<RwaInfoTip text="Limits transfers to whitelisted wallet addresses only. The token issuer manages the whitelist of approved holders." /></span>
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'manage_whitelist')}
                                      className="bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 px-2 py-1 rounded text-xs"
                                      whileTap={{ scale: 0.95 }}
                                    >
                                      Manage Whitelist
                                    </motion.button>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Dividend / Revenue Distribution */}
                          {(contract.features.dividend_enabled || isRealEstate || isEquity || isIP) && (
                            <div className="bg-gradient-to-b from-green-900/10 to-quantum-dark/50 rounded-lg p-4 border border-green-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <DollarSign className="w-4 h-4 text-green-400" />
                                <span className="font-medium text-white">Revenue Distribution</span>
                                <RwaInfoTip text="Distribute revenue (rent, dividends, coupons, royalties) proportionally to all token holders. Amount is split based on ownership percentage." />
                              </div>
                              <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Next Distribution:</span>
                                  <span className="text-green-400 font-medium">Pending</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Total Distributed:</span>
                                  <span className="text-white font-medium">$0.00</span>
                                </div>
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Distribution Amount (USD)<RwaInfoTip text="Total USD amount to distribute. Each holder receives a share proportional to their token ownership (e.g., owning 10% of tokens = 10% of distribution)." /></label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:dividend_amount`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:dividend_amount`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="1000"
                                      className="flex-1 bg-quantum-dark/70 border border-green-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-green-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'distribute_dividend', { amount: getRwaInput(`${contract.address}:dividend_amount`) })}
                                      disabled={!getRwaInput(`${contract.address}:dividend_amount`) || getRwaStatus(contract.address, 'distribute_dividend')?.loading}
                                      className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-50"
                                      whileHover={{ scale: 1.02 }}
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'distribute_dividend')?.loading ? '...' : 'Distribute'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'distribute_dividend') && !getRwaStatus(contract.address, 'distribute_dividend')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'distribute_dividend')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'distribute_dividend')?.message}
                                    </p>
                                  )}
                                </div>
                                <p className="text-xs text-gray-500">Distribute dividends/revenue to all token holders proportionally</p>
                              </div>
                            </div>
                          )}

                          {/* Voting / Governance (Equity specific) */}
                          {(contract.features.voting_rights || isEquity) && (
                            <div className="bg-gradient-to-b from-purple-900/10 to-quantum-dark/50 rounded-lg p-4 border border-purple-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Vote className="w-4 h-4 text-purple-400" />
                                <span className="font-medium text-white">Shareholder Voting</span>
                                <RwaInfoTip text="Create and manage governance proposals for equity token holders. Voting power is proportional to token ownership. Proposals require quorum to pass." />
                              </div>
                              <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Active Proposals:<RwaInfoTip text="Number of currently open proposals. Token holders can vote on active proposals until the voting period ends." /></span>
                                  <span className="text-white font-medium">0</span>
                                </div>
                                <input
                                  type="text"
                                  value={getRwaInput(`${contract.address}:proposal_title`)}
                                  onChange={(e) => setRwaInput(`${contract.address}:proposal_title`, e.target.value)}
                                  placeholder="Proposal title"
                                  className="w-full bg-quantum-dark/70 border border-purple-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-purple-500/50 focus:outline-none"
                                />
                                <textarea
                                  value={getRwaInput(`${contract.address}:proposal_desc`)}
                                  onChange={(e) => setRwaInput(`${contract.address}:proposal_desc`, e.target.value)}
                                  placeholder="Proposal description..."
                                  rows={2}
                                  className="w-full bg-quantum-dark/70 border border-purple-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-purple-500/50 focus:outline-none resize-none"
                                />
                                <motion.button
                                  onClick={() => handleContractAction(contract, 'create_proposal', {
                                    title: getRwaInput(`${contract.address}:proposal_title`),
                                    description: getRwaInput(`${contract.address}:proposal_desc`)
                                  })}
                                  disabled={!getRwaInput(`${contract.address}:proposal_title`) || getRwaStatus(contract.address, 'create_proposal')?.loading}
                                  className="w-full bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                  whileHover={{ scale: 1.02 }}
                                  whileTap={{ scale: 0.98 }}
                                >
                                  {getRwaStatus(contract.address, 'create_proposal')?.loading ? 'Creating...' : 'Create Shareholder Proposal'}
                                </motion.button>
                                {getRwaStatus(contract.address, 'create_proposal') && !getRwaStatus(contract.address, 'create_proposal')?.loading && (
                                  <p className={`text-xs ${getRwaStatus(contract.address, 'create_proposal')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                    {getRwaStatus(contract.address, 'create_proposal')?.message}
                                  </p>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Bond/Fixed Income Controls */}
                          {(contract.features.callable || contract.features.convertible || isFixedIncome) && (
                            <div className="bg-gradient-to-b from-blue-900/10 to-quantum-dark/50 rounded-lg p-4 border border-blue-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Landmark className="w-4 h-4 text-blue-400" />
                                <span className="font-medium text-white">Bond Controls</span>
                                <RwaInfoTip text="Manage fixed-income bond features: pay coupons to holders, call bonds early, or convert to equity if convertible." />
                              </div>
                              <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Credit Rating:<RwaInfoTip text="The bond's credit quality rating (AAA to D). Higher ratings indicate lower default risk and typically offer lower yields." /></span>
                                  <span className="text-blue-400 font-medium">{contract.deploymentParams?.credit_rating || 'N/A'}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Coupon Rate:<RwaInfoTip text="Annual interest rate paid to bondholders, expressed as percentage of face value. Paid on schedule via coupon payments." /></span>
                                  <span className="text-white font-medium">{contract.deploymentParams?.coupon_rate_percent || '0'}%</span>
                                </div>
                                {/* Coupon Payment */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Coupon Amount (USD)</label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:coupon_amount`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:coupon_amount`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="1000"
                                      className="flex-1 bg-quantum-dark/70 border border-blue-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-blue-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'pay_coupon', { amount: getRwaInput(`${contract.address}:coupon_amount`) })}
                                      disabled={!getRwaInput(`${contract.address}:coupon_amount`) || getRwaStatus(contract.address, 'pay_coupon')?.loading}
                                      className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                                      whileHover={{ scale: 1.02 }}
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'pay_coupon')?.loading ? '...' : 'Pay Coupon'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'pay_coupon') && !getRwaStatus(contract.address, 'pay_coupon')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'pay_coupon')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'pay_coupon')?.message}
                                    </p>
                                  )}
                                </div>
                                {contract.features.callable && (
                                  <div>
                                    <label className="text-xs text-gray-400 mb-1 block">Call Price (USD)</label>
                                    <div className="flex gap-2">
                                      <input
                                        type="text"
                                        inputMode="decimal"
                                        value={getRwaInput(`${contract.address}:call_price`)}
                                        onChange={(e) => setRwaInput(`${contract.address}:call_price`, e.target.value.replace(/[^0-9.]/g, ''))}
                                        placeholder="Face value"
                                        className="flex-1 bg-quantum-dark/70 border border-blue-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-blue-500/50 focus:outline-none"
                                      />
                                      <motion.button
                                        onClick={() => handleContractAction(contract, 'call_bond', { call_price: getRwaInput(`${contract.address}:call_price`) })}
                                        disabled={getRwaStatus(contract.address, 'call_bond')?.loading}
                                        className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                                        whileTap={{ scale: 0.98 }}
                                      >
                                        {getRwaStatus(contract.address, 'call_bond')?.loading ? '...' : 'Call Bond'}
                                      </motion.button>
                                    </div>
                                  </div>
                                )}
                                {contract.features.convertible && (
                                  <motion.button
                                    onClick={() => handleContractAction(contract, 'convert_bond')}
                                    disabled={getRwaStatus(contract.address, 'convert_bond')?.loading}
                                    className="w-full bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {getRwaStatus(contract.address, 'convert_bond')?.loading ? 'Converting...' : 'Convert to Equity'}
                                  </motion.button>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Physical Delivery / Redemption */}
                          {(contract.features.delivery_option || contract.features.redemption_enabled || isCommodity || isPhysical) && (
                            <div className="bg-gradient-to-b from-orange-900/10 to-quantum-dark/50 rounded-lg p-4 border border-orange-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Package className="w-4 h-4 text-orange-400" />
                                <span className="font-medium text-white">Delivery & Redemption</span>
                                <RwaInfoTip text="Manage physical delivery of commodity/goods tokens. Token holders can redeem tokens for physical delivery or update storage proofs." />
                              </div>
                              <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Storage:<RwaInfoTip text="Current physical storage facility where the underlying asset is held. Storage conditions are audited and verified on-chain." /></span>
                                  <span className="text-white font-medium">{contract.deploymentParams?.storage_facility || 'N/A'}</span>
                                </div>
                                {/* Process Redemption */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Redemption Quantity</label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:redeem_qty`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:redeem_qty`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="Units to redeem"
                                      className="flex-1 bg-quantum-dark/70 border border-orange-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-orange-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'process_redemption', { quantity: getRwaInput(`${contract.address}:redeem_qty`), request_id: `redeem_${Date.now()}` })}
                                      disabled={!getRwaInput(`${contract.address}:redeem_qty`) || getRwaStatus(contract.address, 'process_redemption')?.loading}
                                      className="bg-gradient-to-r from-orange-600 to-amber-600 hover:from-orange-500 hover:to-amber-500 text-white px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-50"
                                      whileHover={{ scale: 1.02 }}
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'process_redemption')?.loading ? '...' : 'Redeem'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'process_redemption') && !getRwaStatus(contract.address, 'process_redemption')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'process_redemption')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'process_redemption')?.message}
                                    </p>
                                  )}
                                </div>
                                {/* Update Inventory */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Storage Proof / Inventory Update</label>
                                  <input
                                    type="text"
                                    value={getRwaInput(`${contract.address}:storage_proof`)}
                                    onChange={(e) => setRwaInput(`${contract.address}:storage_proof`, e.target.value)}
                                    placeholder="Proof hash or audit reference"
                                    className="w-full bg-quantum-dark/70 border border-orange-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-orange-500/50 focus:outline-none mb-2"
                                  />
                                  <motion.button
                                    onClick={() => handleContractAction(contract, 'update_inventory', { proof_hash: getRwaInput(`${contract.address}:storage_proof`) })}
                                    disabled={getRwaStatus(contract.address, 'update_inventory')?.loading}
                                    className="w-full bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {getRwaStatus(contract.address, 'update_inventory')?.loading ? 'Updating...' : 'Update Inventory / Storage Proof'}
                                  </motion.button>
                                  {getRwaStatus(contract.address, 'update_inventory') && !getRwaStatus(contract.address, 'update_inventory')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_inventory')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_inventory')?.message}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Carbon Credit Controls */}
                          {(contract.features.retirement_enabled || contract.features.offset_tracking || isCarbonCredit) && (
                            <div className="bg-gradient-to-b from-emerald-900/10 to-quantum-dark/50 rounded-lg p-4 border border-emerald-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Leaf className="w-4 h-4 text-emerald-400" />
                                <span className="font-medium text-white">Carbon Credit Management</span>
                                <RwaInfoTip text="Manage carbon offset credits. Retire credits to generate permanent carbon offset certificates. Credits follow Verra VCS or Gold Standard protocols." />
                              </div>
                              <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Verification:<RwaInfoTip text="Carbon credit verification standard used (e.g., Verra VCS, Gold Standard). Determines the quality and market acceptance of the credits." /></span>
                                  <span className="text-emerald-400 font-medium">{contract.deploymentParams?.verification_standard || 'Verra VCS'}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Vintage Year:<RwaInfoTip text="The year the carbon reduction or removal actually occurred. More recent vintages typically command higher prices." /></span>
                                  <span className="text-white font-medium">{contract.deploymentParams?.vintage_year || 'N/A'}</span>
                                </div>
                                {/* Update Verification */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Verifier Name</label>
                                  <input
                                    type="text"
                                    value={getRwaInput(`${contract.address}:verifier`)}
                                    onChange={(e) => setRwaInput(`${contract.address}:verifier`, e.target.value)}
                                    placeholder="e.g., Verra, Gold Standard"
                                    className="w-full bg-quantum-dark/70 border border-emerald-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-emerald-500/50 focus:outline-none mb-2"
                                  />
                                  <motion.button
                                    onClick={() => handleContractAction(contract, 'update_verification', { verifier: getRwaInput(`${contract.address}:verifier`) })}
                                    disabled={getRwaStatus(contract.address, 'update_verification')?.loading}
                                    className="w-full bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {getRwaStatus(contract.address, 'update_verification')?.loading ? 'Updating...' : 'Update Verification Status'}
                                  </motion.button>
                                  {getRwaStatus(contract.address, 'update_verification') && !getRwaStatus(contract.address, 'update_verification')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_verification')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_verification')?.message}
                                    </p>
                                  )}
                                </div>
                                {/* Retire Credits / Issue Certificate */}
                                {contract.features.retirement_enabled && (
                                  <div>
                                    <label className="text-xs text-gray-400 mb-1 block">Tonnes CO2 to Retire</label>
                                    <div className="flex gap-2">
                                      <input
                                        type="text"
                                        inputMode="decimal"
                                        value={getRwaInput(`${contract.address}:retire_tonnes`)}
                                        onChange={(e) => setRwaInput(`${contract.address}:retire_tonnes`, e.target.value.replace(/[^0-9.]/g, ''))}
                                        placeholder="100"
                                        className="flex-1 bg-quantum-dark/70 border border-emerald-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-emerald-500/50 focus:outline-none"
                                      />
                                      <motion.button
                                        onClick={() => handleContractAction(contract, 'retire_credits', { tonnes_co2: getRwaInput(`${contract.address}:retire_tonnes`), beneficiary: getRwaInput(`${contract.address}:beneficiary`) })}
                                        disabled={!getRwaInput(`${contract.address}:retire_tonnes`) || getRwaStatus(contract.address, 'retire_credits')?.loading}
                                        className="bg-gradient-to-r from-emerald-600 to-green-600 hover:from-emerald-500 hover:to-green-500 text-white px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-50"
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                      >
                                        {getRwaStatus(contract.address, 'retire_credits')?.loading ? '...' : 'Retire & Issue Certificate'}
                                      </motion.button>
                                    </div>
                                    <input
                                      type="text"
                                      value={getRwaInput(`${contract.address}:beneficiary`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:beneficiary`, e.target.value)}
                                      placeholder="Beneficiary name (optional)"
                                      className="w-full mt-2 bg-quantum-dark/70 border border-emerald-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-emerald-500/50 focus:outline-none"
                                    />
                                    {getRwaStatus(contract.address, 'retire_credits') && !getRwaStatus(contract.address, 'retire_credits')?.loading && (
                                      <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'retire_credits')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                        {getRwaStatus(contract.address, 'retire_credits')?.message}
                                      </p>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Art & Collectibles Controls */}
                          {(contract.features.provenance_verified || isArt) && (
                            <div className="bg-gradient-to-b from-pink-900/10 to-quantum-dark/50 rounded-lg p-4 border border-pink-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Palette className="w-4 h-4 text-pink-400" />
                                <span className="font-medium text-white">Provenance & Authentication</span>
                                <RwaInfoTip text="Track artwork provenance (ownership history), appraisal values, and custody location. All records are immutably stored on-chain." />
                              </div>
                              <div className="space-y-3">
                                <div className="flex items-center justify-between bg-quantum-dark/40 rounded p-2">
                                  <span className="text-sm text-gray-300">Provenance Verified</span>
                                  <span className={`text-xs font-medium ${contract.features.provenance_verified ? 'text-emerald-400' : 'text-yellow-400'}`}>
                                    {contract.features.provenance_verified ? 'Verified' : 'Pending'}
                                  </span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Current Appraisal:<RwaInfoTip text="Most recent professional appraisal value in USD. Regular appraisals help maintain accurate token pricing and insurance coverage." /></span>
                                  <span className="text-pink-400 font-medium">${Number(contract.deploymentParams?.appraisal_value_usd || 0).toLocaleString()}</span>
                                </div>
                                {/* Update Appraisal */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">New Appraisal Value (USD)</label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:appraisal`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:appraisal`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="500000"
                                      className="flex-1 bg-quantum-dark/70 border border-pink-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-pink-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'update_appraisal', { appraisal_value_usd: getRwaInput(`${contract.address}:appraisal`) })}
                                      disabled={!getRwaInput(`${contract.address}:appraisal`) || getRwaStatus(contract.address, 'update_appraisal')?.loading}
                                      className="bg-pink-500/20 hover:bg-pink-500/30 text-pink-300 px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-50"
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'update_appraisal')?.loading ? '...' : 'Update'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'update_appraisal') && !getRwaStatus(contract.address, 'update_appraisal')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_appraisal')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_appraisal')?.message}
                                    </p>
                                  )}
                                </div>
                                {/* Update Custody */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Custody Location</label>
                                  <input
                                    type="text"
                                    value={getRwaInput(`${contract.address}:custody_location`)}
                                    onChange={(e) => setRwaInput(`${contract.address}:custody_location`, e.target.value)}
                                    placeholder="e.g., Geneva Freeport Vault C-12"
                                    className="w-full bg-quantum-dark/70 border border-pink-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-pink-500/50 focus:outline-none mb-2"
                                  />
                                  <motion.button
                                    onClick={() => handleContractAction(contract, 'update_custody', { custody_location: getRwaInput(`${contract.address}:custody_location`) })}
                                    disabled={getRwaStatus(contract.address, 'update_custody')?.loading}
                                    className="w-full bg-quantum-dark/50 hover:bg-quantum-dark/70 text-gray-200 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {getRwaStatus(contract.address, 'update_custody')?.loading ? 'Updating...' : 'Update Custody Location'}
                                  </motion.button>
                                  {getRwaStatus(contract.address, 'update_custody') && !getRwaStatus(contract.address, 'update_custody')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_custody')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_custody')?.message}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* IP & Royalties Controls */}
                          {(contract.features.sublicensing_allowed || isIP) && (
                            <div className="bg-gradient-to-b from-violet-900/10 to-quantum-dark/50 rounded-lg p-4 border border-violet-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <FileText className="w-4 h-4 text-violet-400" />
                                <span className="font-medium text-white">IP & License Management</span>
                                <RwaInfoTip text="Manage intellectual property licensing, royalty distributions, and sublicensing for IP-backed tokens." />
                              </div>
                              <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">License Type:<RwaInfoTip text="The licensing arrangement type (exclusive, non-exclusive, etc.). Determines sublicensing rights and territory restrictions." /></span>
                                  <span className="text-white font-medium">{contract.deploymentParams?.license_type || 'Exclusive'}</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Royalty Rate:<RwaInfoTip text="Percentage of revenue from IP usage that is distributed to token holders as royalty payments." /></span>
                                  <span className="text-violet-400 font-medium">{contract.deploymentParams?.royalty_rate_percent || '0'}%</span>
                                </div>
                                {/* Distribute Royalties */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Royalty Amount (USD)</label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:royalty_amount`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:royalty_amount`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="5000"
                                      className="flex-1 bg-quantum-dark/70 border border-violet-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-violet-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'distribute_royalties', { amount: getRwaInput(`${contract.address}:royalty_amount`) })}
                                      disabled={!getRwaInput(`${contract.address}:royalty_amount`) || getRwaStatus(contract.address, 'distribute_royalties')?.loading}
                                      className="bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-50"
                                      whileHover={{ scale: 1.02 }}
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'distribute_royalties')?.loading ? '...' : 'Distribute'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'distribute_royalties') && !getRwaStatus(contract.address, 'distribute_royalties')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'distribute_royalties')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'distribute_royalties')?.message}
                                    </p>
                                  )}
                                </div>
                                {/* Manage Sublicenses */}
                                {contract.features.sublicensing_allowed && (
                                  <div>
                                    <label className="text-xs text-gray-400 mb-1 block">Sublicense Details</label>
                                    <input
                                      type="text"
                                      value={getRwaInput(`${contract.address}:licensee`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:licensee`, e.target.value)}
                                      placeholder="Licensee name or address"
                                      className="w-full bg-quantum-dark/70 border border-violet-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-violet-500/50 focus:outline-none mb-2"
                                    />
                                    <input
                                      type="text"
                                      value={getRwaInput(`${contract.address}:license_terms`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:license_terms`, e.target.value)}
                                      placeholder="Terms (e.g., non-exclusive, 2 years)"
                                      className="w-full bg-quantum-dark/70 border border-violet-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-violet-500/50 focus:outline-none mb-2"
                                    />
                                    <motion.button
                                      onClick={() => handleContractAction(contract, 'manage_sublicenses', {
                                        licensee: getRwaInput(`${contract.address}:licensee`),
                                        terms: getRwaInput(`${contract.address}:license_terms`)
                                      })}
                                      disabled={!getRwaInput(`${contract.address}:licensee`) || getRwaStatus(contract.address, 'manage_sublicenses')?.loading}
                                      className="w-full bg-violet-500/20 hover:bg-violet-500/30 text-violet-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                      whileHover={{ scale: 1.02 }}
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'manage_sublicenses')?.loading ? 'Processing...' : 'Grant Sublicense'}
                                    </motion.button>
                                    {getRwaStatus(contract.address, 'manage_sublicenses') && !getRwaStatus(contract.address, 'manage_sublicenses')?.loading && (
                                      <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'manage_sublicenses')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                        {getRwaStatus(contract.address, 'manage_sublicenses')?.message}
                                      </p>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Real Estate Specific */}
                          {isRealEstate && (
                            <div className="bg-gradient-to-b from-amber-900/10 to-quantum-dark/50 rounded-lg p-4 border border-amber-500/10 md:col-span-2">
                              <div className="flex items-center gap-2 mb-3">
                                <Building className="w-4 h-4 text-amber-400" />
                                <span className="font-medium text-white">Property Management</span>
                                <RwaInfoTip text="Manage real estate property details including valuation, occupancy rates, and rental yield. Updates affect token price and yield calculations." />
                              </div>
                              <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                  <div className="bg-quantum-dark/40 rounded p-2">
                                    <span className="text-xs text-gray-500">Location</span>
                                    <div className="text-sm text-white">{contract.deploymentParams?.location || '—'}</div>
                                  </div>
                                  <div className="bg-quantum-dark/40 rounded p-2">
                                    <span className="text-xs text-gray-500">Property Type</span>
                                    <div className="text-sm text-white capitalize">{String(contract.deploymentParams?.property_type || '—').replace(/_/g, ' ')}</div>
                                  </div>
                                  <div className="bg-quantum-dark/40 rounded p-2">
                                    <span className="text-xs text-gray-500">Occupancy Rate<RwaInfoTip text="Percentage of rentable units currently occupied. Higher occupancy = more rental income = higher yield for token holders." position="bottom" /></span>
                                    <div className="text-sm text-amber-400 font-medium">{contract.deploymentParams?.occupancy_rate || '—'}%</div>
                                  </div>
                                  <div className="bg-quantum-dark/40 rounded p-2">
                                    <span className="text-xs text-gray-500">Rental Yield<RwaInfoTip text="Annual rental income as a percentage of property value. This directly determines the dividend rate for token holders." position="bottom" /></span>
                                    <div className="text-sm text-green-400 font-medium">{contract.deploymentParams?.rental_yield_percent || '—'}%</div>
                                  </div>
                                </div>
                                {/* Update Valuation */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">New Valuation (USD)</label>
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      inputMode="numeric"
                                      value={getRwaInput(`${contract.address}:valuation`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:valuation`, e.target.value.replace(/[^0-9]/g, ''))}
                                      placeholder="e.g., 1500000"
                                      className="flex-1 bg-quantum-dark/70 border border-amber-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-amber-500/50 focus:outline-none"
                                    />
                                    <motion.button
                                      onClick={() => {
                                        const val = getRwaInput(`${contract.address}:valuation`);
                                        if (val) handleContractAction(contract, 'update_property_valuation', { valuation_usd: val });
                                      }}
                                      disabled={!getRwaInput(`${contract.address}:valuation`) || getRwaStatus(contract.address, 'update_property_valuation')?.loading}
                                      className="bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                                      whileTap={{ scale: 0.98 }}
                                    >
                                      {getRwaStatus(contract.address, 'update_property_valuation')?.loading ? '...' : 'Update'}
                                    </motion.button>
                                  </div>
                                  {getRwaStatus(contract.address, 'update_property_valuation') && !getRwaStatus(contract.address, 'update_property_valuation')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_property_valuation')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_property_valuation')?.message}
                                    </p>
                                  )}
                                </div>
                                {/* Update Occupancy & Yield */}
                                <div className="grid grid-cols-2 gap-2">
                                  <div>
                                    <label className="text-xs text-gray-400 mb-1 block">Occupancy (%)</label>
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:occupancy`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:occupancy`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="95"
                                      className="w-full bg-quantum-dark/70 border border-amber-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-amber-500/50 focus:outline-none"
                                    />
                                  </div>
                                  <div>
                                    <label className="text-xs text-gray-400 mb-1 block">Rental Yield (%)</label>
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={getRwaInput(`${contract.address}:yield`)}
                                      onChange={(e) => setRwaInput(`${contract.address}:yield`, e.target.value.replace(/[^0-9.]/g, ''))}
                                      placeholder="5.5"
                                      className="w-full bg-quantum-dark/70 border border-amber-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-amber-500/50 focus:outline-none"
                                    />
                                  </div>
                                </div>
                                <motion.button
                                  onClick={() => {
                                    const occ = getRwaInput(`${contract.address}:occupancy`);
                                    const yld = getRwaInput(`${contract.address}:yield`);
                                    handleContractAction(contract, 'update_occupancy', { occupancy_rate: occ || '', rental_yield_percent: yld || '' });
                                  }}
                                  disabled={getRwaStatus(contract.address, 'update_occupancy')?.loading}
                                  className="w-full bg-gradient-to-r from-amber-600 to-yellow-600 hover:from-amber-500 hover:to-yellow-500 text-white px-3 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                                  whileHover={{ scale: 1.02 }}
                                  whileTap={{ scale: 0.98 }}
                                >
                                  {getRwaStatus(contract.address, 'update_occupancy')?.loading ? 'Updating...' : 'Update Occupancy & Yield'}
                                </motion.button>
                                {getRwaStatus(contract.address, 'update_occupancy') && !getRwaStatus(contract.address, 'update_occupancy')?.loading && (
                                  <p className={`text-xs ${getRwaStatus(contract.address, 'update_occupancy')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                    {getRwaStatus(contract.address, 'update_occupancy')?.message}
                                  </p>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Insurance Status (shared across multiple RWA types) */}
                          {(contract.features.insurance_enabled || isCommodity || isArt || isPhysical) && (
                            <div className="bg-gradient-to-b from-cyan-900/10 to-quantum-dark/50 rounded-lg p-4 border border-cyan-500/10">
                              <div className="flex items-center gap-2 mb-3">
                                <Shield className="w-4 h-4 text-cyan-400" />
                                <span className="font-medium text-white">Insurance</span>
                                <RwaInfoTip text="Configure insurance coverage for the underlying asset. Insurance protects token holders against damage, theft, or loss of the physical asset." />
                              </div>
                              <div className="space-y-3">
                                <div className="flex items-center justify-between bg-quantum-dark/40 rounded p-2">
                                  <span className="text-sm text-gray-300">Coverage Status<RwaInfoTip text="Whether insurance coverage is currently active. Inactive coverage means the asset is uninsured and token holders bear full loss risk." /></span>
                                  <span className={`text-xs font-medium ${contract.features.insurance_enabled ? 'text-emerald-400' : 'text-yellow-400'}`}>
                                    {contract.features.insurance_enabled ? 'Active' : 'Inactive'}
                                  </span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span className="text-gray-400">Provider:</span>
                                  <span className="text-white font-medium">{contract.deploymentParams?.insurance_provider || 'None'}</span>
                                </div>
                                {/* Update Insurance */}
                                <div>
                                  <label className="text-xs text-gray-400 mb-1 block">Insurance Provider</label>
                                  <input
                                    type="text"
                                    value={getRwaInput(`${contract.address}:ins_provider`)}
                                    onChange={(e) => setRwaInput(`${contract.address}:ins_provider`, e.target.value)}
                                    placeholder="e.g., Lloyd's of London"
                                    className="w-full bg-quantum-dark/70 border border-cyan-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-cyan-500/50 focus:outline-none mb-2"
                                  />
                                  <label className="text-xs text-gray-400 mb-1 block">Coverage Amount (USD)</label>
                                  <input
                                    type="text"
                                    inputMode="decimal"
                                    value={getRwaInput(`${contract.address}:ins_coverage`)}
                                    onChange={(e) => setRwaInput(`${contract.address}:ins_coverage`, e.target.value.replace(/[^0-9.]/g, ''))}
                                    placeholder="1000000"
                                    className="w-full bg-quantum-dark/70 border border-cyan-500/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-cyan-500/50 focus:outline-none mb-2"
                                  />
                                  <motion.button
                                    onClick={() => handleContractAction(contract, 'update_insurance', {
                                      provider: getRwaInput(`${contract.address}:ins_provider`),
                                      coverage_usd: getRwaInput(`${contract.address}:ins_coverage`)
                                    })}
                                    disabled={getRwaStatus(contract.address, 'update_insurance')?.loading}
                                    className="w-full bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 px-3 py-2 rounded-lg text-sm font-bold transition-colors disabled:opacity-50"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {getRwaStatus(contract.address, 'update_insurance')?.loading ? 'Updating...' : 'Update Insurance Policy'}
                                  </motion.button>
                                  {getRwaStatus(contract.address, 'update_insurance') && !getRwaStatus(contract.address, 'update_insurance')?.loading && (
                                    <p className={`text-xs mt-1 ${getRwaStatus(contract.address, 'update_insurance')?.success ? 'text-green-400' : 'text-red-400'}`}>
                                      {getRwaStatus(contract.address, 'update_insurance')?.message}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      );
                    })()}

                    {/* View in Explorer */}
                    <motion.button
                      onClick={() => {
                        // Navigate to explorer
                        console.log('View in explorer:', contract.address);
                      }}
                      className="w-full bg-quantum-dark/50 hover:bg-quantum-dark/70 text-quantum-cyan px-4 py-3 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                    >
                      <FileCode className="w-4 h-4" />
                      View in Explorer
                    </motion.button>
                  </div>
                    </motion.div>
                  )}

                  {/* Events Tab */}
                  {getActiveTab(contract.address) === 'events' && (
                    <motion.div
                      key="events"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-4"
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <History className="w-4 h-4 text-quantum-cyan" />
                        <h4 className="font-bold text-white">Event History</h4>
                      </div>

                      {/* Event List */}
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {getContractEvents(contract).length === 0 ? (
                          <div className="bg-quantum-dark/50 rounded-lg p-6 text-center">
                            <Activity className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                            <p className="text-gray-400 text-sm">No events recorded yet</p>
                          </div>
                        ) : (
                          getContractEvents(contract).map((event) => (
                            <div
                              key={event.id}
                              className="bg-quantum-dark/50 rounded-lg p-4 border border-quantum-purple/10 hover:border-quantum-purple/30 transition-colors"
                            >
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  {event.type === 'mint' && <ArrowUpRight className="w-4 h-4 text-quantum-green" />}
                                  {event.type === 'burn' && <Flame className="w-4 h-4 text-quantum-orange" />}
                                  {event.type === 'transfer' && <Send className="w-4 h-4 text-quantum-blue" />}
                                  {event.type === 'airdrop' && <Gift className="w-4 h-4 text-quantum-purple" />}
                                  {event.type === 'pause' && <PauseCircle className="w-4 h-4 text-quantum-yellow" />}
                                  {event.type === 'unpause' && <PlayCircle className="w-4 h-4 text-quantum-green" />}
                                  {event.type === 'stake' && <TrendingUp className="w-4 h-4 text-quantum-cyan" />}
                                  {event.type === 'unstake' && <ArrowDownRight className="w-4 h-4 text-quantum-pink" />}
                                  {event.type === 'reflection' && <Percent className="w-4 h-4 text-quantum-purple" />}
                                  <span className="font-medium text-white capitalize">{event.type}</span>
                                </div>
                                <div className="flex items-center gap-1 text-xs text-gray-500">
                                  <Clock className="w-3 h-3" />
                                  {event.timestamp.toLocaleDateString()} {event.timestamp.toLocaleTimeString()}
                                </div>
                              </div>

                              {event.amount && (
                                <div className="flex items-center gap-2 text-sm mb-2">
                                  <span className="text-gray-400">Amount:</span>
                                  <span className={`font-medium ${
                                    event.type === 'mint' ? 'text-quantum-green' :
                                    event.type === 'burn' ? 'text-quantum-orange' :
                                    'text-white'
                                  }`}>
                                    {event.type === 'mint' ? '+' : event.type === 'burn' ? '-' : ''}{formatDisplayAmount(event.amount)} {contract.symbol}
                                  </span>
                                </div>
                              )}

                              {event.to && (
                                <div className="flex items-center gap-2 text-sm mb-2">
                                  <span className="text-gray-400">To:</span>
                                  <span className="font-mono text-xs text-quantum-cyan truncate max-w-xs">{event.to}</span>
                                </div>
                              )}

                              {event.recipients && (
                                <div className="flex items-center gap-2 text-sm mb-2">
                                  <span className="text-gray-400">Recipients:</span>
                                  <span className="text-white">{event.recipients} addresses</span>
                                </div>
                              )}

                              <div className="flex items-center gap-2 text-xs">
                                <span className="text-gray-500">TX:</span>
                                <span className="font-mono text-quantum-green/70 truncate">{event.txHash}</span>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </motion.div>
                  )}

                  {/* Stats Tab */}
                  {getActiveTab(contract.address) === 'stats' && (
                    <motion.div
                      key="stats"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-4"
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <PieChart className="w-4 h-4 text-quantum-green" />
                        <h4 className="font-bold text-white">Token Statistics</h4>
                      </div>

                      {/* Supply Stats */}
                      <div className="bg-quantum-dark/50 rounded-lg p-4">
                        <h5 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                          <Coins className="w-4 h-4" />
                          Supply Information
                        </h5>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Total Supply</div>
                            <div className="text-lg font-bold text-white">{getContractStats(contract).totalSupply}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Circulating</div>
                            <div className="text-lg font-bold text-quantum-cyan">{getContractStats(contract).circulatingSupply}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Burned</div>
                            <div className="text-lg font-bold text-quantum-orange">{getContractStats(contract).burnedTokens}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Holders</div>
                            <div className="text-lg font-bold text-quantum-purple">{getContractStats(contract).holders}</div>
                          </div>
                        </div>
                      </div>

                      {/* Activity Stats */}
                      <div className="bg-quantum-dark/50 rounded-lg p-4">
                        <h5 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                          <Activity className="w-4 h-4" />
                          Activity
                        </h5>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Total Minted</div>
                            <div className="text-lg font-bold text-quantum-green">{getContractStats(contract).totalMinted}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Total Burned</div>
                            <div className="text-lg font-bold text-quantum-orange">{getContractStats(contract).totalBurned}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Total Airdropped</div>
                            <div className="text-lg font-bold text-quantum-blue">{getContractStats(contract).totalAirdropped}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 mb-1">Total Transfers</div>
                            <div className="text-lg font-bold text-white">{getContractStats(contract).totalTransfers}</div>
                          </div>
                        </div>
                      </div>

                      {/* Staking Stats (if applicable) */}
                      {contract.features.staking && (
                        <div className="bg-gradient-to-r from-quantum-cyan/10 to-quantum-blue/10 border border-quantum-cyan/30 rounded-lg p-4">
                          <h5 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-quantum-cyan" />
                            Staking
                          </h5>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <div className="text-xs text-gray-500 mb-1">APY</div>
                              <div className="text-lg font-bold text-quantum-green">{getContractStats(contract).stakingAPY}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500 mb-1">Total Staked</div>
                              <div className="text-lg font-bold text-quantum-cyan">{getContractStats(contract).totalStaked}</div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Reflection Stats (if applicable) */}
                      {contract.features.reflection && (
                        <div className="bg-gradient-to-r from-quantum-purple/10 to-quantum-pink/10 border border-quantum-purple/30 rounded-lg p-4">
                          <h5 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                            <Percent className="w-4 h-4 text-quantum-purple" />
                            Reflections
                          </h5>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <div className="text-xs text-gray-500 mb-1">Reflection Rate</div>
                              <div className="text-lg font-bold text-quantum-purple">{getContractStats(contract).reflectionRate}</div>
                            </div>
                            <div>
                              <div className="text-xs text-gray-500 mb-1">Total Distributed</div>
                              <div className="text-lg font-bold text-quantum-pink">{getContractStats(contract).totalReflections}</div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* View in Explorer */}
                      <motion.button
                        onClick={() => {
                          console.log('View in explorer:', contract.address);
                        }}
                        className="w-full bg-quantum-dark/50 hover:bg-quantum-dark/70 text-quantum-cyan px-4 py-3 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                      >
                        <FileCode className="w-4 h-4" />
                        View Full Analytics in Explorer
                      </motion.button>
                    </motion.div>
                  )}

                  {/* Social Tab - v2.4.8 */}
                  {getActiveTab(contract.address) === 'social' && (
                    <motion.div
                      key="social"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-4"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4 text-quantum-pink" />
                          <h4 className="font-bold text-white">Social Media & Links</h4>
                        </div>
                        {editingSocial !== contract.address && (
                          <motion.button
                            onClick={() => {
                              setEditingSocial(contract.address);
                              setSocialFormData(socialProfiles[contract.address] || {});
                            }}
                            className="text-sm text-quantum-cyan hover:text-white transition-colors"
                            whileHover={{ scale: 1.05 }}
                          >
                            Edit Profile
                          </motion.button>
                        )}
                      </div>

                      {/* Edit Mode */}
                      {editingSocial === contract.address ? (
                        <div className="space-y-3">
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">X / Twitter</label>
                            <input
                              type="text"
                              value={socialFormData.twitter || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, twitter: e.target.value }))}
                              placeholder="@username or https://x.com/..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-pink/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">Discord</label>
                            <input
                              type="text"
                              value={socialFormData.discord || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, discord: e.target.value }))}
                              placeholder="https://discord.gg/..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-purple/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">Telegram</label>
                            <input
                              type="text"
                              value={socialFormData.telegram || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, telegram: e.target.value }))}
                              placeholder="https://t.me/..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-blue/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">Website</label>
                            <input
                              type="text"
                              value={socialFormData.website || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, website: e.target.value }))}
                              placeholder="https://..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">GitHub</label>
                            <input
                              type="text"
                              value={socialFormData.github || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, github: e.target.value }))}
                              placeholder="https://github.com/..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-green/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">Medium / Blog</label>
                            <input
                              type="text"
                              value={socialFormData.medium || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, medium: e.target.value }))}
                              placeholder="https://medium.com/..."
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-orange/50 focus:outline-none"
                            />
                          </div>
                          <div className="bg-quantum-dark/50 rounded-lg p-4">
                            <label className="text-xs text-gray-400 mb-1 block">Description</label>
                            <textarea
                              value={socialFormData.description || ''}
                              onChange={(e) => setSocialFormData(prev => ({ ...prev, description: e.target.value }))}
                              placeholder="Describe your token project..."
                              rows={3}
                              className="w-full bg-quantum-dark/70 border border-quantum-purple/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:border-quantum-pink/50 focus:outline-none resize-none"
                            />
                          </div>
                          <div className="flex gap-2">
                            <motion.button
                              onClick={() => setEditingSocial(null)}
                              className="flex-1 bg-quantum-dark/50 hover:bg-quantum-dark/70 text-gray-400 px-4 py-2 rounded-lg text-sm font-medium"
                              whileHover={{ scale: 1.02 }}
                            >
                              Cancel
                            </motion.button>
                            <motion.button
                              onClick={() => handleSaveSocial(contract)}
                              disabled={savingSocial}
                              className="flex-1 bg-gradient-to-r from-quantum-pink to-quantum-purple hover:from-quantum-pink/80 hover:to-quantum-purple/80 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
                              whileHover={{ scale: 1.02 }}
                            >
                              {savingSocial ? 'Saving...' : 'Save Profile'}
                            </motion.button>
                          </div>
                        </div>
                      ) : (
                        /* View Mode */
                        <div className="space-y-3">
                          {/* Social Links Display */}
                          {socialProfiles[contract.address] && Object.values(socialProfiles[contract.address]).some(v => v) ? (
                            <>
                              {socialProfiles[contract.address]?.description && (
                                <div className="bg-quantum-dark/50 rounded-lg p-4">
                                  <p className="text-gray-300 text-sm">{socialProfiles[contract.address].description}</p>
                                </div>
                              )}
                              <div className="grid grid-cols-2 gap-3">
                                {socialProfiles[contract.address]?.twitter && (
                                  <a
                                    href={socialProfiles[contract.address].twitter!.startsWith('http')
                                      ? socialProfiles[contract.address].twitter
                                      : `https://x.com/${socialProfiles[contract.address].twitter!.replace('@', '')}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-dark/70 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">𝕏</span>
                                    <span className="text-sm truncate">{socialProfiles[contract.address].twitter}</span>
                                  </a>
                                )}
                                {socialProfiles[contract.address]?.discord && (
                                  <a
                                    href={socialProfiles[contract.address].discord}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-purple/30 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">💬</span>
                                    <span className="text-sm">Discord</span>
                                  </a>
                                )}
                                {socialProfiles[contract.address]?.telegram && (
                                  <a
                                    href={socialProfiles[contract.address].telegram}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-blue/30 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">✈️</span>
                                    <span className="text-sm">Telegram</span>
                                  </a>
                                )}
                                {socialProfiles[contract.address]?.website && (
                                  <a
                                    href={socialProfiles[contract.address].website}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-cyan/30 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">🌐</span>
                                    <span className="text-sm">Website</span>
                                  </a>
                                )}
                                {socialProfiles[contract.address]?.github && (
                                  <a
                                    href={socialProfiles[contract.address].github}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-green/30 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">⚙️</span>
                                    <span className="text-sm">GitHub</span>
                                  </a>
                                )}
                                {socialProfiles[contract.address]?.medium && (
                                  <a
                                    href={socialProfiles[contract.address].medium}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="bg-quantum-dark/50 hover:bg-quantum-orange/30 rounded-lg p-3 flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
                                  >
                                    <span className="text-lg">📝</span>
                                    <span className="text-sm">Medium</span>
                                  </a>
                                )}
                              </div>
                            </>
                          ) : (
                            <div className="bg-quantum-dark/50 rounded-lg p-6 text-center">
                              <Users className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                              <p className="text-gray-400 text-sm mb-3">No social links added yet</p>
                              <motion.button
                                onClick={() => {
                                  setEditingSocial(contract.address);
                                  setSocialFormData({});
                                }}
                                className="text-sm text-quantum-pink hover:text-white transition-colors"
                                whileHover={{ scale: 1.05 }}
                              >
                                Add Social Links
                              </motion.button>
                            </div>
                          )}

                          {/* Creator Trust Score */}
                          <div className="bg-gradient-to-r from-quantum-pink/10 to-quantum-purple/10 border border-quantum-pink/30 rounded-lg p-4">
                            <h5 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-quantum-pink" />
                              Creator Reputation
                            </h5>
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Trust Score:</span>
                                <span className="font-bold text-quantum-green">85/100</span>
                              </div>
                              <div className="w-full bg-quantum-dark/50 rounded-full h-2">
                                <div className="bg-gradient-to-r from-quantum-green to-quantum-cyan h-2 rounded-full" style={{ width: '85%' }}></div>
                              </div>
                              <div className="flex justify-between text-xs text-gray-500 mt-2">
                                <span>First token created</span>
                                <span>No rug history ✓</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
