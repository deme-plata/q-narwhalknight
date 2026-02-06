import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Code, Coins, Building, Vote, Lock, ArrowRight, Sparkles, CheckCircle, FileCode, Settings, Flame, Zap, Users, PauseCircle, PlayCircle, RefreshCw, Upload, Send, History, BarChart3, TrendingUp, Activity, Clock, ArrowUpRight, ArrowDownRight, Gift, Percent, PieChart } from 'lucide-react';

type ContractCategory = 'tokens' | 'defi' | 'rwa' | 'governance';
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
  };
  isPaused?: boolean;
  logoUrl?: string; // libp2p IPFS CID for logo
  logoDataUrl?: string; // Base64 data URL for display (temporary until uploaded)
  abaBalance?: string; // User's balance of this token
  totalSupply?: string; // v1.4.9: Total supply of the token (for event history)
  decimals?: number; // v1.4.9: Token decimals for balance conversion
  owner?: string; // v2.4.8: Contract owner/creator address
  socialMedia?: SocialMediaProfile; // v2.4.8: Social links
}

interface ContractTemplate {
  id: string;
  name: string;
  description: string;
  icon: typeof Coins;
  category: ContractCategory;
  features: string[];
  gasEstimate: string;
}

// Dynamic gas pricing: Target $0.10 USD equivalent in QUG
// As QUG price increases, gas cost in QUG decreases proportionally
// This ensures fees stay affordable even if 1 QUG = $1M in the future
const TARGET_FEE_USD = 0.10; // Target $0.10 USD per contract deployment

// Calculate dynamic gas based on USD target and oracle price
const calculateDynamicGas = (baseUsdCost: number, qugPriceUsd: number): number => {
  // If QUG = $1M, then 0.0000001 QUG = $0.10
  // If QUG = $0.01, then 10 QUG = $0.10
  const qugAmount = baseUsdCost / qugPriceUsd;
  return Math.max(0.0000001, qugAmount); // Minimum 0.0000001 QUG (100 nanoQUG)
};

const contractTemplates: ContractTemplate[] = [
  {
    id: 'secure-token',
    name: 'Secure Token',
    description: 'Basic quantum-safe token with transfer and balance tracking',
    icon: Coins,
    category: 'tokens',
    features: ['Transfer', 'Balance Tracking', 'Quantum-Safe'],
    gasEstimate: '~$0.10 USD' // Will be calculated dynamically from oracle
  },
  {
    id: 'advanced-token',
    name: 'Advanced Token',
    description: 'Feature-rich token with minting, burning, staking, and governance',
    icon: Sparkles,
    category: 'tokens',
    features: ['Mintable', 'Burnable', 'Staking', 'Governance', 'Reflection', 'Pausable'],
    gasEstimate: '~$0.20 USD' // Will be calculated dynamically from oracle
  },
  {
    id: 'rwa-token',
    name: 'RWA Token',
    description: 'Real-world asset tokenization with compliance features',
    icon: Building,
    category: 'rwa',
    features: ['Asset Backing', 'Compliance', 'KYC Integration', 'Transfer Restrictions'],
    gasEstimate: '~$0.30 USD' // Will be calculated dynamically from oracle
  },
  {
    id: 'governance',
    name: 'Governance DAO',
    description: 'Decentralized governance with proposal and voting mechanisms',
    icon: Vote,
    category: 'governance',
    features: ['Proposals', 'Voting', 'Timelock', 'Delegation'],
    gasEstimate: '~$0.40 USD' // Will be calculated dynamically from oracle
  },
  {
    id: 'private-dex',
    name: 'Private DEX',
    description: 'Privacy-preserving decentralized exchange with ZK-SNARKs',
    icon: Lock,
    category: 'defi',
    features: ['Private Swaps', 'ZK-SNARKs', 'Liquidity Pools', 'AMM'],
    gasEstimate: '~$0.50 USD' // Will be calculated dynamically from oracle
  }
];

const categories = [
  { id: 'tokens' as ContractCategory, name: 'Tokens', icon: Coins },
  { id: 'defi' as ContractCategory, name: 'DeFi', icon: Lock },
  { id: 'rwa' as ContractCategory, name: 'RWA', icon: Building },
  { id: 'governance' as ContractCategory, name: 'Governance', icon: Vote }
];

export default function VittuaVMScreen() {
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

  // Contract control states
  const [mintAmount, setMintAmount] = useState('');
  const [burnAmount, setBurnAmount] = useState('');
  const [reflectionRate, setReflectionRate] = useState('2');
  const [airdropAddresses, setAirdropAddresses] = useState('');
  const [airdropAmount, setAirdropAmount] = useState('');

  const toggleFeature = (feature: keyof typeof features) => {
    setFeatures(prev => ({ ...prev, [feature]: !prev[feature] }));
  };

  const filteredTemplates = contractTemplates.filter(t => t.category === selectedCategory);

  const handleSelectTemplate = (template: ContractTemplate) => {
    setSelectedTemplate(template);
    // Advanced Token goes to features step, others go to basics
    if (template.id === 'advanced-token') {
      setStep('basics');
    } else {
      setStep('basics');
    }
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
          initialSupply: initialSupplyBaseUnits, // v3.2.18-beta: Send BASE units
          decimals: tokenDecimals, // v3.2.18-beta: User-selected decimals
          logoIpfsCid: logoIpfsCid || undefined,
          ...features, // Include all feature flags
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

      // Call the contract deployment API
      const deployResponse = await fetch('/api/v1/contracts/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(deploymentRequest),
      });

      if (!deployResponse.ok) {
        const error = await deployResponse.json();
        throw new Error(error.error || 'Failed to deploy contract via API');
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
      };

      const updatedContracts = [...deployedContracts, newContract];
      setDeployedContracts(updatedContracts);

      // Update localStorage cache immediately (walletAddress already defined above)
      const cacheKey = `deployedContracts_${walletAddress}`;
      localStorage.setItem(cacheKey, JSON.stringify(updatedContracts));
      console.log('💾 Updated cache with newly deployed contract');

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

      <AnimatePresence mode="wait">
        {/* Step 1: Contract Selection */}
        {step === 'select' && (
          <motion.div
            key="select"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {/* Category Tabs */}
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
                  onClick={() => selectedTemplate.id === 'advanced-token' ? setStep('features') : setStep('review')}
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

                <div className="bg-gradient-to-r from-quantum-yellow/10 to-quantum-orange/10 border border-quantum-yellow/30 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-300">Estimated Gas Cost</div>
                    <div className="text-lg font-bold text-quantum-yellow">{selectedTemplate.gasEstimate}</div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3">
                <motion.button
                  onClick={() => setStep(selectedTemplate.id === 'advanced-token' ? 'features' : 'basics')}
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

      {/* My Deployed Contracts Section */}
      {step === 'select' && (loadingContracts || deployedContracts.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          <h2 className="text-2xl font-bold text-white">My Deployed Contracts</h2>

          {loadingContracts && (
            <div className="bg-quantum-indigo/30 backdrop-blur-xl border border-quantum-purple/30 rounded-xl p-6 text-center">
              <p className="text-gray-400">Loading deployed contracts from blockchain...</p>
            </div>
          )}

          <div className="space-y-4">
            {deployedContracts.map((contract) => (
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
                  {getActiveTab(contract.address) === 'control' && Object.keys(contract.features).length > 0 && (
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
