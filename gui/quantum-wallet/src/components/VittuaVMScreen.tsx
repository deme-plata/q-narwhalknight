import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Code, Coins, Building, Vote, Lock, ArrowRight, Sparkles, CheckCircle, FileCode, Settings, Flame, Zap, Users, PauseCircle, PlayCircle, RefreshCw, Upload } from 'lucide-react';

type ContractCategory = 'tokens' | 'defi' | 'rwa' | 'governance';
type DeploymentStep = 'select' | 'basics' | 'features' | 'review' | 'deploying' | 'success';

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
  };
  isPaused?: boolean;
  logoUrl?: string; // libp2p IPFS CID for logo
  logoDataUrl?: string; // Base64 data URL for display (temporary until uploaded)
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
    upgradeable: false
  });

  // Deployed contracts - fetched from API
  // Load deployed contracts from localStorage
  const [deployedContracts, setDeployedContracts] = useState<DeployedContract[]>(() => {
    const saved = localStorage.getItem('deployedContracts');
    return saved ? JSON.parse(saved) : [];
  });

  // Save deployed contracts to localStorage whenever they change
  React.useEffect(() => {
    localStorage.setItem('deployedContracts', JSON.stringify(deployedContracts));
  }, [deployedContracts]);

  // Contract control states
  const [mintAmount, setMintAmount] = useState('');
  const [burnAmount, setBurnAmount] = useState('');
  const [reflectionRate, setReflectionRate] = useState('2');

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

      // Sign transaction via wallet endpoint
      const zeroAddress = new Array(32).fill(0); // Contract deployment address (all zeros)

      const signRequest = {
        to: zeroAddress,
        amount: 0, // No QUG transfer, just deployment
        fee: fee,
        password: '', // In production, prompt for password
        data: metadataBytes, // Contract metadata in transaction data
      };

      console.log('✍️  Signing contract deployment transaction...');
      console.log('📋 Using wallet ID:', walletId);

      const signResponse = await fetch(`/api/v1/wallets/${walletId}/sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signRequest),
      });

      if (!signResponse.ok) {
        const error = await signResponse.json();
        throw new Error(error.error || 'Failed to sign transaction');
      }

      const signResult = await signResponse.json();
      console.log('📦 Sign response:', signResult);

      // Validate the response structure
      if (!signResult || !signResult.success) {
        throw new Error(signResult?.error || 'Transaction signing failed');
      }

      if (!signResult.data) {
        throw new Error('No transaction data returned from signing');
      }

      const signedTx = signResult.data;

      // Validate signed transaction has required fields
      if (!signedTx.id) {
        console.error('Invalid signed transaction:', signedTx);
        throw new Error('Signed transaction is missing ID field');
      }

      console.log('✅ Transaction signed:', signedTx.id);
      console.log('📡 Broadcasting to blockchain...');

      // Submit transaction to blockchain
      const submitResponse = await fetch('/api/v1/transactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction: signedTx }),
      });

      if (!submitResponse.ok) {
        const error = await submitResponse.json();
        throw new Error(error.error || 'Failed to submit transaction');
      }

      const submitResult = await submitResponse.json();
      console.log('✅ Contract deployed on blockchain!', submitResult);

      // Convert transaction ID (byte array) to hex string with qnk prefix
      const txIdArray = Array.isArray(signedTx.id) ? signedTx.id : Object.values(signedTx.id);
      const contractAddress = 'qnk' + txIdArray.map((b: number) => b.toString(16).padStart(2, '0')).join('');

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
      };

      setDeployedContracts(prev => [...prev, newContract]);
      setStep('success');

    } catch (error: any) {
      console.error('❌ Contract deployment failed:', error);
      alert(`Deployment failed: ${error.message}`);
      setStep('review'); // Go back to review step on error
    }
  };

  const handleMint = (contract: DeployedContract) => {
    console.log(`Minting ${mintAmount} ${contract.symbol}`);
    setMintAmount('');
  };

  const handleBurn = (contract: DeployedContract) => {
    console.log(`Burning ${burnAmount} ${contract.symbol}`);
    setBurnAmount('');
  };

  const handleTogglePause = (contract: DeployedContract) => {
    setDeployedContracts(prev =>
      prev.map(c =>
        c.address === contract.address
          ? { ...c, isPaused: !c.isPaused }
          : c
      )
    );
  };

  const handleUpdateReflection = (contract: DeployedContract) => {
    console.log(`Setting reflection rate to ${reflectionRate}% for ${contract.symbol}`);
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

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Initial Supply
                      </label>
                      <input
                        type="number"
                        value={initialSupply}
                        onChange={(e) => setInitialSupply(e.target.value)}
                        placeholder="1000000"
                        className="w-full bg-quantum-dark/50 border border-quantum-cyan/20 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-quantum-cyan/50 focus:outline-none"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Total tokens to mint at deployment
                      </p>
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
                    <div className="text-lg font-bold text-white">{Number(initialSupply).toLocaleString()} {tokenSymbol}</div>
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
                  qnk{Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)}
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
      {step === 'select' && deployedContracts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          <h2 className="text-2xl font-bold text-white">My Deployed Contracts</h2>

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

                {/* Contract Controls */}
                {Object.keys(contract.features).length > 0 && (
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
                              type="number"
                              value={mintAmount}
                              onChange={(e) => setMintAmount(e.target.value)}
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
                              type="number"
                              value={burnAmount}
                              onChange={(e) => setBurnAmount(e.target.value)}
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
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
