// RWA Template Data - Extracted for testability
// This file contains all RWA template definitions, field configurations,
// contract type mappings, and gas fee multipliers used by VittuaVMScreen.tsx

import { Building, TrendingUp, Landmark, Gem, Leaf, Palette, FileText, Package, Coins, Sparkles, Vote, Lock } from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────

export type ContractCategory = 'tokens' | 'defi' | 'rwa' | 'governance' | 'derivatives';

export interface ContractTemplate {
  id: string;
  name: string;
  description: string;
  icon: typeof Coins;
  category: ContractCategory;
  features: string[];
  gasEstimate: string;
}

export interface RwaField {
  key: string;
  label: string;
  type: 'text' | 'select' | 'checkbox' | 'number';
  placeholder?: string;
  options?: { value: string; label: string }[];
  description?: string;
  section?: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

// Dynamic gas pricing: Target $0.10 USD equivalent in QUG
export const TARGET_FEE_USD = 0.10;

// Calculate dynamic gas based on USD target and oracle price
export const calculateDynamicGas = (baseUsdCost: number, qugPriceUsd: number): number => {
  const qugAmount = baseUsdCost / qugPriceUsd;
  return Math.max(0.0000001, qugAmount);
};

export const contractTemplates: ContractTemplate[] = [
  {
    id: 'secure-token',
    name: 'Secure Token',
    description: 'Basic quantum-safe token with transfer and balance tracking',
    icon: Coins,
    category: 'tokens',
    features: ['Transfer', 'Balance Tracking', 'Quantum-Safe'],
    gasEstimate: '~$0.10 USD'
  },
  {
    id: 'advanced-token',
    name: 'Advanced Token',
    description: 'Feature-rich token with minting, burning, staking, and governance',
    icon: Sparkles,
    category: 'tokens',
    features: ['Mintable', 'Burnable', 'Staking', 'Governance', 'Reflection', 'Pausable'],
    gasEstimate: '~$0.20 USD'
  },
  {
    id: 'rwa-token',
    name: 'RWA Token',
    description: 'Real-world asset tokenization with compliance features',
    icon: Building,
    category: 'rwa',
    features: ['Asset Backing', 'Compliance', 'KYC Integration', 'Transfer Restrictions'],
    gasEstimate: '~$0.30 USD'
  },
  {
    id: 'real-estate-token',
    name: 'Real Estate',
    description: 'Tokenize residential, commercial, or industrial property with rental yields and fractional ownership',
    icon: Building,
    category: 'rwa',
    features: ['Property Backing', 'Rental Yields', 'Fractional Ownership', 'KYC/AML', 'Dividend Distribution'],
    gasEstimate: '~$0.40 USD'
  },
  {
    id: 'equity-token',
    name: 'Equity & Shares',
    description: 'Tokenize company equity with voting rights, dividends, and vesting schedules',
    icon: TrendingUp,
    category: 'rwa',
    features: ['Voting Rights', 'Dividends', 'Vesting', 'Share Classes', 'Board Governance'],
    gasEstimate: '~$0.45 USD'
  },
  {
    id: 'fixed-income-token',
    name: 'Fixed Income',
    description: 'Bonds, treasury notes, and debt instruments with coupon payments and maturity dates',
    icon: Landmark,
    category: 'rwa',
    features: ['Coupon Payments', 'Maturity Date', 'Credit Rating', 'Callable', 'Convertible'],
    gasEstimate: '~$0.40 USD'
  },
  {
    id: 'commodity-token',
    name: 'Commodities',
    description: 'Gold, oil, agriculture - tokenized commodities with storage proof and optional physical delivery',
    icon: Gem,
    category: 'rwa',
    features: ['Storage Proof', 'Physical Delivery', 'Spot Price Oracle', 'Insurance', 'Quantity Tracking'],
    gasEstimate: '~$0.35 USD'
  },
  {
    id: 'carbon-credit-token',
    name: 'Carbon Credits',
    description: 'Verified carbon offset credits with retirement tracking and environmental impact reporting',
    icon: Leaf,
    category: 'rwa',
    features: ['Credit Retirement', 'Verification', 'Impact Tracking', 'Offset Certificates', 'Project Linking'],
    gasEstimate: '~$0.30 USD'
  },
  {
    id: 'art-collectible-token',
    name: 'Art & Collectibles',
    description: 'Fractional ownership of fine art, rare collectibles, wine, watches, and memorabilia',
    icon: Palette,
    category: 'rwa',
    features: ['Provenance Chain', 'Appraisal History', 'Fractional Ownership', 'Insurance', 'Authentication'],
    gasEstimate: '~$0.35 USD'
  },
  {
    id: 'ip-revenue-token',
    name: 'IP & Royalties',
    description: 'Tokenize patents, copyrights, trademarks, and royalty revenue streams',
    icon: FileText,
    category: 'rwa',
    features: ['Revenue Sharing', 'License Management', 'Royalty Distribution', 'IP Registration', 'Sublicensing'],
    gasEstimate: '~$0.40 USD'
  },
  {
    id: 'physical-goods-token',
    name: 'Physical Goods',
    description: 'Luxury goods, electronics, vehicles - with supply chain tracking and redemption',
    icon: Package,
    category: 'rwa',
    features: ['Supply Chain Proof', 'Redemption', 'Inventory Tracking', 'Shipping', 'Serial Numbers'],
    gasEstimate: '~$0.35 USD'
  },
  {
    id: 'governance',
    name: 'Governance DAO',
    description: 'Decentralized governance with proposal and voting mechanisms',
    icon: Vote,
    category: 'governance',
    features: ['Proposals', 'Voting', 'Timelock', 'Delegation'],
    gasEstimate: '~$0.40 USD'
  },
  {
    id: 'private-dex',
    name: 'Private DEX',
    description: 'Privacy-preserving decentralized exchange with ZK-SNARKs',
    icon: Lock,
    category: 'defi',
    features: ['Private Swaps', 'ZK-SNARKs', 'Liquidity Pools', 'AMM'],
    gasEstimate: '~$0.50 USD'
  }
];

// RWA-specific deployment parameters per template type
export const rwaFieldsByTemplate: Record<string, RwaField[]> = {
  'real-estate-token': [
    { key: 'property_type', label: 'Property Type', type: 'select', section: 'Property Details', options: [
      { value: 'residential', label: 'Residential' }, { value: 'commercial', label: 'Commercial' },
      { value: 'industrial', label: 'Industrial' }, { value: 'land', label: 'Land' }, { value: 'mixed_use', label: 'Mixed Use' },
    ]},
    { key: 'location', label: 'Property Location', type: 'text', section: 'Property Details', placeholder: 'e.g., Manhattan, NY' },
    { key: 'total_valuation_usd', label: 'Total Valuation (USD)', type: 'number', section: 'Property Details', placeholder: '1000000' },
    { key: 'property_area_sqft', label: 'Property Area (sqft)', type: 'number', section: 'Property Details', placeholder: '2500' },
    { key: 'rental_yield_percent', label: 'Rental Yield (%)', type: 'number', section: 'Income & Returns', placeholder: '5.5' },
    { key: 'occupancy_rate', label: 'Occupancy Rate (%)', type: 'number', section: 'Income & Returns', placeholder: '95' },
    { key: 'kyc_required', label: 'KYC Required', type: 'checkbox', section: 'Compliance', description: 'Require KYC/AML verification for investors' },
    { key: 'accredited_only', label: 'Accredited Investors Only', type: 'checkbox', section: 'Compliance', description: 'Restrict to accredited investors' },
    { key: 'dividend_enabled', label: 'Enable Dividends', type: 'checkbox', section: 'Compliance', description: 'Distribute rental income to token holders' },
    { key: 'transfer_restrictions', label: 'Transfer Restrictions', type: 'checkbox', section: 'Compliance', description: 'Restrict transfers for regulatory compliance' },
  ],
  'equity-token': [
    { key: 'share_class', label: 'Share Class', type: 'select', section: 'Share Structure', options: [
      { value: 'common', label: 'Common Shares' }, { value: 'preferred', label: 'Preferred Shares' }, { value: 'restricted', label: 'Restricted Stock' },
    ]},
    { key: 'company_name', label: 'Company Name', type: 'text', section: 'Share Structure', placeholder: 'Acme Corp' },
    { key: 'dividend_schedule', label: 'Dividend Schedule', type: 'select', section: 'Dividends & Governance', options: [
      { value: 'quarterly', label: 'Quarterly' }, { value: 'semi_annual', label: 'Semi-Annual' }, { value: 'annual', label: 'Annual' }, { value: 'none', label: 'None' },
    ]},
    { key: 'vesting_period_months', label: 'Vesting Period (months)', type: 'number', section: 'Vesting & Lock-Up', placeholder: '12' },
    { key: 'lockup_period_months', label: 'Lock-Up Period (months)', type: 'number', section: 'Vesting & Lock-Up', placeholder: '6' },
    { key: 'board_seats_per_share', label: 'Board Seats Per Share', type: 'number', section: 'Dividends & Governance', placeholder: '0' },
    { key: 'voting_rights', label: 'Voting Rights', type: 'checkbox', section: 'Dividends & Governance', description: 'Token holders can vote on proposals' },
    { key: 'kyc_required', label: 'KYC Required', type: 'checkbox', section: 'Compliance', description: 'Require KYC/AML for investors' },
    { key: 'accredited_only', label: 'Accredited Investors Only', type: 'checkbox', section: 'Compliance', description: 'Restrict to accredited investors' },
  ],
  'fixed-income-token': [
    { key: 'instrument_type', label: 'Instrument Type', type: 'select', section: 'Bond Details', options: [
      { value: 'corporate_bond', label: 'Corporate Bond' }, { value: 'treasury_note', label: 'Treasury Note' },
      { value: 'municipal_bond', label: 'Municipal Bond' }, { value: 'convertible_note', label: 'Convertible Note' },
    ]},
    { key: 'face_value_usd', label: 'Face Value (USD)', type: 'number', section: 'Bond Details', placeholder: '1000' },
    { key: 'coupon_rate_percent', label: 'Coupon Rate (%)', type: 'number', section: 'Bond Details', placeholder: '5.0' },
    { key: 'maturity_date', label: 'Maturity Date', type: 'text', section: 'Bond Details', placeholder: '2030-12-31' },
    { key: 'payment_frequency', label: 'Payment Frequency', type: 'select', section: 'Payment Schedule', options: [
      { value: 'monthly', label: 'Monthly' }, { value: 'quarterly', label: 'Quarterly' },
      { value: 'semi_annual', label: 'Semi-Annual' }, { value: 'annual', label: 'Annual' },
    ]},
    { key: 'credit_rating', label: 'Credit Rating', type: 'select', section: 'Rating & Risk', options: [
      { value: 'AAA', label: 'AAA' }, { value: 'AA', label: 'AA' }, { value: 'A', label: 'A' },
      { value: 'BBB', label: 'BBB' }, { value: 'BB', label: 'BB' }, { value: 'B', label: 'B' },
    ]},
    { key: 'callable', label: 'Callable', type: 'checkbox', section: 'Rating & Risk', description: 'Issuer can redeem before maturity' },
    { key: 'convertible', label: 'Convertible', type: 'checkbox', section: 'Rating & Risk', description: 'Can convert to equity tokens' },
  ],
  'commodity-token': [
    { key: 'commodity_type', label: 'Commodity Type', type: 'select', section: 'Commodity Details', options: [
      { value: 'precious_metal', label: 'Precious Metal' }, { value: 'energy', label: 'Energy' },
      { value: 'agriculture', label: 'Agriculture' }, { value: 'industrial_metal', label: 'Industrial Metal' },
    ]},
    { key: 'unit_of_measurement', label: 'Unit of Measurement', type: 'select', section: 'Commodity Details', options: [
      { value: 'troy_oz', label: 'Troy Ounce' }, { value: 'kg', label: 'Kilogram' },
      { value: 'barrel', label: 'Barrel' }, { value: 'bushel', label: 'Bushel' }, { value: 'mwh', label: 'MWh' },
    ]},
    { key: 'quantity_per_token', label: 'Quantity Per Token', type: 'number', section: 'Commodity Details', placeholder: '1' },
    { key: 'storage_provider', label: 'Storage Provider', type: 'text', section: 'Storage & Delivery', placeholder: 'e.g., Brinks, Loomis' },
    { key: 'storage_location', label: 'Storage Location', type: 'text', section: 'Storage & Delivery', placeholder: 'e.g., London, Zurich' },
    { key: 'delivery_option', label: 'Physical Delivery', type: 'checkbox', section: 'Storage & Delivery', description: 'Allow physical delivery/redemption' },
    { key: 'insurance_enabled', label: 'Insurance', type: 'checkbox', section: 'Storage & Delivery', description: 'Asset is insured against loss' },
    { key: 'spot_price_oracle', label: 'Spot Price Oracle', type: 'text', section: 'Pricing', placeholder: 'e.g., Chainlink XAU/USD' },
  ],
  'carbon-credit-token': [
    { key: 'credit_standard', label: 'Credit Standard', type: 'select', section: 'Credit Details', options: [
      { value: 'verra_vcs', label: 'Verra VCS' }, { value: 'gold_standard', label: 'Gold Standard' },
      { value: 'acr', label: 'ACR' }, { value: 'car', label: 'CAR' },
    ]},
    { key: 'project_type', label: 'Project Type', type: 'select', section: 'Credit Details', options: [
      { value: 'reforestation', label: 'Reforestation' }, { value: 'renewable_energy', label: 'Renewable Energy' },
      { value: 'methane_capture', label: 'Methane Capture' }, { value: 'direct_air_capture', label: 'Direct Air Capture' },
      { value: 'ocean_conservation', label: 'Ocean Conservation' },
    ]},
    { key: 'vintage_year', label: 'Vintage Year', type: 'number', section: 'Credit Details', placeholder: '2025' },
    { key: 'total_credits_tonnes', label: 'Total Credits (Tonnes CO2)', type: 'number', section: 'Credit Details', placeholder: '10000' },
    { key: 'verification_body', label: 'Verification Body', type: 'text', section: 'Verification', placeholder: 'e.g., SCS Global' },
    { key: 'project_location', label: 'Project Location', type: 'text', section: 'Verification', placeholder: 'e.g., Amazon Basin, Brazil' },
    { key: 'retirement_enabled', label: 'Retirement Enabled', type: 'checkbox', section: 'Features', description: 'Allow permanent credit retirement (offset)' },
    { key: 'offset_tracking', label: 'Offset Tracking', type: 'checkbox', section: 'Features', description: 'Track and publish offset certificates' },
  ],
  'art-collectible-token': [
    { key: 'item_type', label: 'Item Type', type: 'select', section: 'Item Details', options: [
      { value: 'painting', label: 'Painting' }, { value: 'sculpture', label: 'Sculpture' },
      { value: 'digital_art', label: 'Digital Art' }, { value: 'photography', label: 'Photography' },
      { value: 'rare_collectible', label: 'Rare Collectible' }, { value: 'wine_spirits', label: 'Wine & Spirits' },
    ]},
    { key: 'artist_creator', label: 'Artist / Creator', type: 'text', section: 'Item Details', placeholder: 'e.g., Pablo Picasso' },
    { key: 'creation_year', label: 'Creation Year', type: 'number', section: 'Item Details', placeholder: '2024' },
    { key: 'appraisal_value_usd', label: 'Appraisal Value (USD)', type: 'number', section: 'Valuation', placeholder: '500000' },
    { key: 'total_fractions', label: 'Total Fractions', type: 'number', section: 'Valuation', placeholder: '1000' },
    { key: 'physical_custody', label: 'Physical Custody', type: 'select', section: 'Custody', options: [
      { value: 'owner', label: 'Owner' }, { value: 'vault', label: 'Secure Vault' },
      { value: 'museum', label: 'Museum' }, { value: 'gallery', label: 'Gallery' },
    ]},
    { key: 'provenance_verified', label: 'Provenance Verified', type: 'checkbox', section: 'Authentication', description: 'Provenance chain has been verified' },
    { key: 'insurance_enabled', label: 'Insured', type: 'checkbox', section: 'Authentication', description: 'Item is professionally insured' },
  ],
  'ip-revenue-token': [
    { key: 'ip_type', label: 'IP Type', type: 'select', section: 'IP Details', options: [
      { value: 'patent', label: 'Patent' }, { value: 'copyright', label: 'Copyright' },
      { value: 'trademark', label: 'Trademark' }, { value: 'music_royalty', label: 'Music Royalty' },
      { value: 'film_royalty', label: 'Film Royalty' }, { value: 'software_license', label: 'Software License' },
    ]},
    { key: 'jurisdiction', label: 'Jurisdiction', type: 'text', section: 'IP Details', placeholder: 'e.g., United States' },
    { key: 'registration_number', label: 'Registration Number', type: 'text', section: 'IP Details', placeholder: 'e.g., US-PAT-12345678' },
    { key: 'revenue_share_percent', label: 'Revenue Share (%)', type: 'number', section: 'Revenue', placeholder: '10' },
    { key: 'expiry_date', label: 'IP Expiry Date', type: 'text', section: 'Revenue', placeholder: '2045-01-01' },
    { key: 'distribution_frequency', label: 'Revenue Distribution', type: 'select', section: 'Revenue', options: [
      { value: 'monthly', label: 'Monthly' }, { value: 'quarterly', label: 'Quarterly' }, { value: 'annual', label: 'Annual' },
    ]},
    { key: 'minimum_guarantee_usd', label: 'Minimum Guarantee (USD)', type: 'number', section: 'Revenue', placeholder: '0' },
    { key: 'sublicensing_allowed', label: 'Sublicensing Allowed', type: 'checkbox', section: 'License Terms', description: 'Allow sublicensing of the IP' },
  ],
  'physical-goods-token': [
    { key: 'product_category', label: 'Product Category', type: 'select', section: 'Product Details', options: [
      { value: 'luxury_goods', label: 'Luxury Goods' }, { value: 'electronics', label: 'Electronics' },
      { value: 'automotive', label: 'Automotive' }, { value: 'jewelry', label: 'Jewelry' },
      { value: 'watches', label: 'Watches' }, { value: 'other', label: 'Other' },
    ]},
    { key: 'manufacturer', label: 'Manufacturer / Brand', type: 'text', section: 'Product Details', placeholder: 'e.g., Rolex, Tesla' },
    { key: 'warehouse_location', label: 'Warehouse Location', type: 'text', section: 'Logistics', placeholder: 'e.g., Singapore Freeport' },
    { key: 'serial_number_tracking', label: 'Serial Number Tracking', type: 'checkbox', section: 'Logistics', description: 'Track items by serial number' },
    { key: 'supply_chain_verified', label: 'Supply Chain Verified', type: 'checkbox', section: 'Logistics', description: 'Full supply chain provenance verified' },
    { key: 'redemption_enabled', label: 'Physical Redemption', type: 'checkbox', section: 'Redemption', description: 'Token holders can redeem for physical item' },
    { key: 'shipping_included', label: 'Shipping Included', type: 'checkbox', section: 'Redemption', description: 'Shipping costs included in token price' },
    { key: 'insurance_enabled', label: 'Insurance', type: 'checkbox', section: 'Redemption', description: 'Items are insured during storage and shipping' },
  ],
};

// Map frontend template ID to backend contract type
export const contractTypeMap: Record<string, string> = {
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

// Gas fee multipliers per template
// Template multipliers: secure-token=1x, advanced-token=2x, rwa-token=3x, governance=4x, private-dex=5x
export const feeMultipliers: Record<string, number> = {
  'secure-token': 1,
  'advanced-token': 2,
  'rwa-token': 3,
  'real-estate-token': 4,
  'equity-token': 4,
  'fixed-income-token': 3.5,
  'commodity-token': 3.5,
  'carbon-credit-token': 3,
  'art-collectible-token': 3.5,
  'ip-revenue-token': 3.5,
  'physical-goods-token': 3.5,
  'governance': 4,
  'private-dex': 5,
};

// The 8 specialized RWA template IDs (excluding the generic 'rwa-token')
export const RWA_SPECIALIZED_TEMPLATE_IDS = [
  'real-estate-token',
  'equity-token',
  'fixed-income-token',
  'commodity-token',
  'carbon-credit-token',
  'art-collectible-token',
  'ip-revenue-token',
  'physical-goods-token',
] as const;

// All RWA template IDs (including the generic 'rwa-token')
export const RWA_ALL_TEMPLATE_IDS = [
  'rwa-token',
  ...RWA_SPECIALIZED_TEMPLATE_IDS,
] as const;

export const categories = [
  { id: 'tokens' as ContractCategory, name: 'Tokens', icon: Coins },
  { id: 'defi' as ContractCategory, name: 'DeFi', icon: Lock },
  { id: 'rwa' as ContractCategory, name: 'RWA', icon: Building },
  { id: 'governance' as ContractCategory, name: 'Governance', icon: Vote },
];
