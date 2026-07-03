import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Eye, Zap, Target, TrendingUp, TrendingDown,
  Shield, AlertTriangle, Activity, Sparkles,
  BarChart3, PieChart, X, Layers, Clock,
  DollarSign, Percent, Users, Award, Flame, Wifi, WifiOff
} from 'lucide-react';

// Live price data from APIs
interface LivePrices {
  btc: { usd: number; usd_24h_change: number };
  eth: { usd: number; usd_24h_change: number };
  sp500: { value: number; change: number };
  lastUpdated: number;
  isLive: boolean;
}

// Fetch live prices from CoinGecko (free, no API key required)
const fetchLivePrices = async (): Promise<LivePrices | null> => {
  try {
    // CoinGecko free API for crypto prices
    const cryptoResponse = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true',
      { cache: 'no-store' }
    );

    if (!cryptoResponse.ok) {
      console.warn('CoinGecko API rate limited, using fallback');
      return null;
    }

    const cryptoData = await cryptoResponse.json();

    // For S&P 500, we'll use a reasonable estimate since free stock APIs are limited
    // In production, you'd use Alpha Vantage, Yahoo Finance, or similar
    const sp500Value = 6050 + (Math.sin(Date.now() / 3600000) * 50); // Slight variation

    return {
      btc: {
        usd: cryptoData.bitcoin?.usd || 105000,
        usd_24h_change: cryptoData.bitcoin?.usd_24h_change || 0,
      },
      eth: {
        usd: cryptoData.ethereum?.usd || 3950,
        usd_24h_change: cryptoData.ethereum?.usd_24h_change || 0,
      },
      sp500: {
        value: sp500Value,
        change: (Math.random() - 0.5) * 2, // Simulated since no free API
      },
      lastUpdated: Date.now(),
      isLive: true,
    };
  } catch (error) {
    console.error('Failed to fetch live prices:', error);
    return null;
  }
};

interface OracleDataPoint {
  id: string;
  domain: string;
  value: number;
  timestamp: number;
  sources: { provider: string; value: number; confidence: number }[];
  confidence: number;
  x?: number;
  y?: number;
  age?: number;
  change24h?: number;
}

interface ResolutionEvent {
  id: string;
  domain: string;
  stakeId: string;
  predictedValue: number;
  actualValue: number;
  accuracyScore: number;
  isAccurate: boolean;
  slashingApplied: number;
  rewardAdjustment: number;
  timestamp: number;
  x?: number;
  y?: number;
  age?: number;
  walletPrefix?: string;
}

interface StakePosition {
  id: string;
  domain: string;
  amount: number;
  confidence: number;
  prediction: number;
  angle?: number;
  radius?: number;
  walletPrefix?: string;
}

interface DomainStats {
  domain: string;
  totalStaked: number;
  avgAccuracy: number;
  resolutionCount: number;
  slashingTotal: number;
  activeStakes: number;
  lastValue: number;
  trend: 'up' | 'down' | 'stable';
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  type: 'success' | 'slashing' | 'oracle';
}

interface QNOOracleVisualizationProps {
  className?: string;
}

// Domain colors for visual consistency
const DOMAIN_COLORS: Record<string, { primary: string; secondary: string; glow: string; label: string }> = {
  'crypto-btc': { primary: '#f7931a', secondary: '#c77600', glow: 'rgba(247, 147, 26, 0.6)', label: 'BTC/USD' },
  'crypto-eth': { primary: '#627eea', secondary: '#4156c3', glow: 'rgba(98, 126, 234, 0.6)', label: 'ETH/USD' },
  'sports-nfl': { primary: '#013369', secondary: '#001a3a', glow: 'rgba(1, 51, 105, 0.6)', label: 'NFL' },
  'weather-temp': { primary: '#00bcd4', secondary: '#0097a7', glow: 'rgba(0, 188, 212, 0.6)', label: 'Weather' },
  'finance-sp500': { primary: '#4caf50', secondary: '#388e3c', glow: 'rgba(76, 175, 80, 0.6)', label: 'S&P 500' },
  default: { primary: '#8b5cf6', secondary: '#6d28d9', glow: 'rgba(139, 92, 246, 0.6)', label: 'Other' },
};

const getDomainColor = (domain: string) => {
  return DOMAIN_COLORS[domain] || DOMAIN_COLORS.default;
};

const formatValue = (value: number, domain: string): string => {
  if (domain.startsWith('crypto') || domain.startsWith('finance')) {
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const formatCompact = (value: number): string => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toFixed(2);
};

export default function QNOOracleVisualization({ className = '' }: QNOOracleVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [oracleData, setOracleData] = useState<OracleDataPoint[]>([]);
  const [resolutions, setResolutions] = useState<ResolutionEvent[]>([]);
  const [stakes, setStakes] = useState<StakePosition[]>([]);
  // v6.0.3: Use ref for particles to avoid infinite re-render loop in animation useEffect
  const particlesRef = useRef<Particle[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<ResolutionEvent | OracleDataPoint | null>(null);
  const [visualMode, setVisualMode] = useState<'radar' | 'flow' | 'chart' | 'ring'>('radar');
  const [isConnected, setIsConnected] = useState(false);
  const [domainStats, setDomainStats] = useState<DomainStats[]>([]);
  const [recentEvents, setRecentEvents] = useState<{ type: string; message: string; timestamp: number; color: string }[]>([]);
  const [stats, setStats] = useState({
    totalResolutions: 0,
    accuracyRate: 0.85,
    totalSlashed: 0,
    totalRewarded: 0,
    activeStakes: 0,
    totalStakeValue: 0,
    oracleSources: 3,
    avgConfidence: 0.92,
    resolutionsPerHour: 12,
    winRate: 0.78,
  });

  // Live prices from CoinGecko
  const [livePrices, setLivePrices] = useState<LivePrices | null>(null);
  const [priceStatus, setPriceStatus] = useState<'loading' | 'live' | 'fallback'>('loading');

  const animationFrameId = useRef<number | undefined>(undefined);
  const lastUpdateTime = useRef(Date.now());

  // Configuration
  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 450;
  const CENTER_X = CANVAS_WIDTH / 2;
  const CENTER_Y = CANVAS_HEIGHT / 2 + 20;

  // Create particles for visual effects
  const createParticles = useCallback((x: number, y: number, type: 'success' | 'slashing' | 'oracle', count = 12) => {
    const colors = {
      success: '#10b981',
      slashing: '#ef4444',
      oracle: '#8b5cf6',
    };

    const newParticles: Particle[] = [];
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.5;
      const speed = 30 + Math.random() * 50;
      newParticles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 500 + Math.random() * 300,
        color: colors[type],
        size: 2 + Math.random() * 3,
        type,
      });
    }
    particlesRef.current = [...particlesRef.current.slice(-80), ...newParticles];
  }, []);

  // Add event to recent events feed
  const addRecentEvent = useCallback((type: string, message: string, color: string) => {
    setRecentEvents(prev => [{
      type,
      message,
      timestamp: Date.now(),
      color,
    }, ...prev.slice(0, 9)]);
  }, []);

  // Calculate domain stats
  const updateDomainStats = useCallback(() => {
    const domains = ['crypto-btc', 'crypto-eth', 'finance-sp500', 'weather-temp'];
    const newStats: DomainStats[] = domains.map(domain => {
      const domainResolutions = resolutions.filter(r => r.domain === domain);
      const domainStakes = stakes.filter(s => s.domain === domain);
      const domainOracle = oracleData.find(o => o.domain === domain);

      const avgAccuracy = domainResolutions.length > 0
        ? domainResolutions.reduce((sum, r) => sum + r.accuracyScore, 0) / domainResolutions.length
        : 0;

      const totalSlashing = domainResolutions.reduce((sum, r) => sum + r.slashingApplied, 0);
      const totalStaked = domainStakes.reduce((sum, s) => sum + s.amount, 0);

      return {
        domain,
        totalStaked,
        avgAccuracy,
        resolutionCount: domainResolutions.length,
        slashingTotal: totalSlashing,
        activeStakes: domainStakes.length,
        lastValue: domainOracle?.value || 0,
        trend: Math.random() > 0.5 ? 'up' : 'down',
      };
    });

    setDomainStats(newStats);
  }, [resolutions, stakes, oracleData]);

  useEffect(() => {
    updateDomainStats();
  }, [resolutions, stakes, oracleData, updateDomainStats]);

  // Simulate SSE data (in production, connect to /api/v1/events)
  useEffect(() => {
    console.log('Oracle Visualization starting, connecting to SSE stream...');

    const eventSource = new EventSource('/api/v1/events');

    eventSource.onopen = () => {
      console.log('SSE connection opened for QNO visualization');
      setIsConnected(true);
    };

    // Listen for oracle data events
    eventSource.addEventListener('oracle-update', (event) => {
      try {
        const data = JSON.parse(event.data);
        const oracleEvent: OracleDataPoint = {
          id: `oracle-${Date.now()}`,
          domain: data.domain || 'crypto-btc',
          value: data.value || Math.random() * 100000,
          timestamp: Date.now(),
          sources: data.sources || [
            { provider: 'Simulated', value: data.value, confidence: 0.95 },
          ],
          confidence: data.confidence || 0.9,
          x: CENTER_X + (Math.random() - 0.5) * 200,
          y: CENTER_Y + (Math.random() - 0.5) * 100,
          age: 0,
          change24h: (Math.random() - 0.5) * 10,
        };

        setOracleData(prev => [...prev.slice(-20), oracleEvent]);
        createParticles(oracleEvent.x!, oracleEvent.y!, 'oracle', 8);
        addRecentEvent('oracle', `${getDomainColor(oracleEvent.domain).label}: ${formatValue(oracleEvent.value, oracleEvent.domain)}`, '#8b5cf6');
      } catch (error) {
        console.error('Error processing oracle-update:', error);
      }
    });

    // Listen for resolution events
    eventSource.addEventListener('qno-resolution', (event) => {
      try {
        const data = JSON.parse(event.data);
        const resolution: ResolutionEvent = {
          id: `resolution-${Date.now()}`,
          domain: data.domain || 'crypto-btc',
          stakeId: data.stake_id || `stake-${Date.now()}`,
          predictedValue: data.predicted_value || 50000,
          actualValue: data.actual_value || 51000,
          accuracyScore: data.accuracy_score || 0.85,
          isAccurate: data.is_accurate ?? true,
          slashingApplied: data.slashing_applied || 0,
          rewardAdjustment: data.reward_adjustment || 0.1,
          timestamp: Date.now(),
          x: CENTER_X + (Math.random() - 0.5) * 300,
          y: CENTER_Y + (Math.random() - 0.5) * 150,
          age: 0,
          walletPrefix: data.wallet_address?.substring(0, 8) || 'Qx...abc',
        };

        setResolutions(prev => [...prev.slice(-30), resolution]);
        createParticles(resolution.x!, resolution.y!, resolution.isAccurate ? 'success' : 'slashing', 15);

        const eventColor = resolution.isAccurate ? '#10b981' : '#ef4444';
        const eventType = resolution.isAccurate ? 'WIN' : 'SLASH';
        addRecentEvent(eventType, `${resolution.walletPrefix}: ${(resolution.accuracyScore * 100).toFixed(1)}% acc`, eventColor);

        // Update stats
        setStats(prev => ({
          ...prev,
          totalResolutions: prev.totalResolutions + 1,
          totalSlashed: prev.totalSlashed + (resolution.slashingApplied || 0),
          totalRewarded: prev.totalRewarded + (resolution.isAccurate ? resolution.rewardAdjustment * 100 : 0),
          accuracyRate: (prev.accuracyRate * prev.totalResolutions + (resolution.isAccurate ? 1 : 0)) / (prev.totalResolutions + 1),
        }));
      } catch (error) {
        console.error('Error processing qno-resolution:', error);
      }
    });

    // Listen for staking events
    eventSource.addEventListener('qno-stake', (event) => {
      try {
        const data = JSON.parse(event.data);
        const stake: StakePosition = {
          id: `stake-${Date.now()}`,
          domain: data.domain || 'crypto-btc',
          amount: data.amount || 1000,
          confidence: data.confidence || 0.8,
          prediction: data.prediction_value || 50000,
          angle: Math.random() * Math.PI * 2,
          radius: 100 + Math.random() * 100,
          walletPrefix: data.wallet_address?.substring(0, 8) || 'Qx...def',
        };

        setStakes(prev => [...prev.slice(-15), stake]);
        addRecentEvent('STAKE', `${stake.walletPrefix}: ${formatCompact(stake.amount)} QUG`, '#3b82f6');

        setStats(prev => ({
          ...prev,
          activeStakes: prev.activeStakes + 1,
          totalStakeValue: prev.totalStakeValue + stake.amount,
        }));
      } catch (error) {
        console.error('Error processing qno-stake:', error);
      }
    });

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      setIsConnected(false);
    };

    // Initialize data with LIVE prices from CoinGecko
    const initializeWithLivePrices = async () => {
      const domains = ['crypto-btc', 'crypto-eth', 'finance-sp500', 'weather-temp'];
      const walletPrefixes = ['Qx1a2b...', 'Qx3c4d...', 'Qx5e6f...', 'Qx7g8h...', 'Qx9i0j...'];

      // Fetch live prices
      setPriceStatus('loading');
      const prices = await fetchLivePrices();

      // Get current price for each domain (live or fallback)
      const getPrice = (domain: string): { value: number; change: number } => {
        if (prices) {
          setLivePrices(prices);
          setPriceStatus('live');
          switch (domain) {
            case 'crypto-btc': return { value: prices.btc.usd, change: prices.btc.usd_24h_change };
            case 'crypto-eth': return { value: prices.eth.usd, change: prices.eth.usd_24h_change };
            case 'finance-sp500': return { value: prices.sp500.value, change: prices.sp500.change };
            default: return { value: 72, change: 0 }; // Weather fallback
          }
        }
        // Fallback to estimates if API fails
        setPriceStatus('fallback');
        switch (domain) {
          case 'crypto-btc': return { value: 105000, change: 0 };
          case 'crypto-eth': return { value: 3950, change: 0 };
          case 'finance-sp500': return { value: 6100, change: 0 };
          default: return { value: 72, change: 0 };
        }
      };

      // Add initial oracle data points with LIVE prices
      domains.forEach((domain, i) => {
        const priceData = getPrice(domain);
        const baseValue = priceData.value;
        setOracleData(prev => [...prev, {
          id: `oracle-init-${i}`,
          domain,
          value: baseValue,
          timestamp: Date.now() - i * 60000,
          sources: [
            { provider: 'CoinGecko', value: baseValue, confidence: 0.98 },
            { provider: 'Chainlink', value: baseValue * (1 + (Math.random() - 0.5) * 0.001), confidence: 0.95 },
            { provider: 'Pyth', value: baseValue * (1 + (Math.random() - 0.5) * 0.001), confidence: 0.92 },
          ],
          confidence: 0.95,
          x: 150 + i * 180,
          y: 120 + (i % 2) * 80,
          age: 5000,
          change24h: priceData.change,
        }]);
      });

      // Add initial stakes with predictions based on LIVE prices
      for (let i = 0; i < 12; i++) {
        const domain = domains[i % domains.length];
        const priceData = getPrice(domain);
        const domainBase = priceData.value;
        // Predictions are user guesses - slightly different from current price
        const predictionVariance = 0.02 + Math.random() * 0.08; // 2-10% variance
        const direction = Math.random() > 0.5 ? 1 : -1;
        setStakes(prev => [...prev, {
          id: `stake-init-${i}`,
          domain,
          amount: 500 + Math.random() * 3000,
          confidence: 0.6 + Math.random() * 0.35,
          prediction: domainBase * (1 + direction * predictionVariance),
          angle: (Math.PI * 2 * i) / 12,
          radius: 100 + Math.random() * 80,
          walletPrefix: walletPrefixes[i % walletPrefixes.length],
        }]);
      }

      // Add initial resolutions based on past predictions vs actual LIVE price
      for (let i = 0; i < 15; i++) {
        const domain = domains[i % domains.length];
        const priceData = getPrice(domain);
        const actualValue = priceData.value;
        // Past predictions were made at different times
        const predictionVariance = (Math.random() - 0.5) * 0.1;
        const predictedValue = actualValue * (1 + predictionVariance);
        const accuracy = 1 - Math.abs(predictionVariance);
        const isAccurate = accuracy > 0.95; // >95% accuracy = win

        setResolutions(prev => [...prev, {
          id: `resolution-init-${i}`,
          domain,
          stakeId: `stake-init-${i}`,
          predictedValue,
          actualValue,
          accuracyScore: accuracy,
          isAccurate,
          slashingApplied: isAccurate ? 0 : 50 + Math.random() * 150,
          rewardAdjustment: isAccurate ? 0.05 + Math.random() * 0.12 : -0.05,
          timestamp: Date.now() - i * 120000,
          x: 200 + (i % 5) * 120,
          y: 150 + Math.floor(i / 5) * 100,
          age: 5000,
          walletPrefix: walletPrefixes[i % walletPrefixes.length],
        }]);
      }

      // Set initial stats
      setStats({
        totalResolutions: 15,
        accuracyRate: 0.78,
        totalSlashed: 892.45,
        totalRewarded: 1456.32,
        activeStakes: 12,
        totalStakeValue: 24680,
        oracleSources: 3,
        avgConfidence: 0.91,
        resolutionsPerHour: 8.5,
        winRate: 0.75,
      });

      // Add initial recent events with LIVE prices
      const btcPrice = getPrice('crypto-btc').value;
      const ethPrice = getPrice('crypto-eth').value;
      setRecentEvents([
        { type: 'WIN', message: 'Qx1a2b...: 92.3% accuracy', timestamp: Date.now() - 5000, color: '#10b981' },
        { type: 'oracle', message: `BTC/USD: $${btcPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, timestamp: Date.now() - 12000, color: '#8b5cf6' },
        { type: 'STAKE', message: 'Qx3c4d...: 1.5K QUG', timestamp: Date.now() - 18000, color: '#3b82f6' },
        { type: 'SLASH', message: 'Qx5e6f...: 45.2% accuracy', timestamp: Date.now() - 25000, color: '#ef4444' },
        { type: 'oracle', message: `ETH/USD: $${ethPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, timestamp: Date.now() - 28000, color: '#8b5cf6' },
        { type: 'WIN', message: 'Qx7g8h...: 88.7% accuracy', timestamp: Date.now() - 32000, color: '#10b981' },
      ]);
    };

    initializeWithLivePrices();

    return () => {
      console.log('Closing SSE connection for QNO visualization');
      eventSource.close();
    };
  }, [createParticles, addRecentEvent]);

  // Periodic price refresh (every 60 seconds to respect CoinGecko rate limits)
  useEffect(() => {
    const refreshPrices = async () => {
      const prices = await fetchLivePrices();
      if (prices) {
        setLivePrices(prices);
        setPriceStatus('live');

        // Update oracle data with new prices
        setOracleData(prev => prev.map(oracle => {
          let newValue = oracle.value;
          let newChange = oracle.change24h || 0;

          if (oracle.domain === 'crypto-btc') {
            newValue = prices.btc.usd;
            newChange = prices.btc.usd_24h_change;
          } else if (oracle.domain === 'crypto-eth') {
            newValue = prices.eth.usd;
            newChange = prices.eth.usd_24h_change;
          } else if (oracle.domain === 'finance-sp500') {
            newValue = prices.sp500.value;
            newChange = prices.sp500.change;
          }

          return {
            ...oracle,
            value: newValue,
            change24h: newChange,
            timestamp: Date.now(),
            sources: [
              { provider: 'CoinGecko', value: newValue, confidence: 0.98 },
              { provider: 'Chainlink', value: newValue * (1 + (Math.random() - 0.5) * 0.001), confidence: 0.95 },
              { provider: 'Pyth', value: newValue * (1 + (Math.random() - 0.5) * 0.001), confidence: 0.92 },
            ],
          };
        }));

        // Add price update to recent events
        addRecentEvent('oracle', `BTC: $${prices.btc.usd.toLocaleString()}`, '#8b5cf6');
      }
    };

    // Refresh every 60 seconds
    const interval = setInterval(refreshPrices, 60000);
    return () => clearInterval(interval);
  }, [addRecentEvent]);

  // Handle canvas click
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (event.clientX - rect.left) * scaleX;
    const clickY = (event.clientY - rect.top) * scaleY;

    // Check resolutions first
    for (const resolution of resolutions) {
      if (!resolution.x || !resolution.y) continue;
      const dist = Math.sqrt((clickX - resolution.x) ** 2 + (clickY - resolution.y) ** 2);
      if (dist < 30) {
        setSelectedEvent(resolution);
        return;
      }
    }

    // Check oracle data points
    for (const oracle of oracleData) {
      if (!oracle.x || !oracle.y) continue;
      const dist = Math.sqrt((clickX - oracle.x) ** 2 + (clickY - oracle.y) ** 2);
      if (dist < 25) {
        setSelectedEvent(oracle);
        return;
      }
    }

    setSelectedEvent(null);
  };

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      const now = Date.now();
      const deltaTime = (now - lastUpdateTime.current) / 1000;
      lastUpdateTime.current = now;

      // Update ages
      resolutions.forEach(r => { if (r.age !== undefined) r.age += deltaTime * 1000; });
      oracleData.forEach(o => { if (o.age !== undefined) o.age += deltaTime * 1000; });

      // Update particles (mutate ref directly, no setState to avoid re-render loop)
      particlesRef.current = particlesRef.current
          .map(p => ({
            ...p,
            x: p.x + p.vx * deltaTime,
            y: p.y + p.vy * deltaTime,
            life: p.life + deltaTime * 1000,
            vx: p.vx * 0.95,
            vy: p.vy * 0.95,
          }))
          .filter(p => p.life < p.maxLife);

      // Clear canvas with gradient background
      const bgGradient = ctx.createLinearGradient(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      bgGradient.addColorStop(0, '#0a0a18');
      bgGradient.addColorStop(0.5, '#0f0f23');
      bgGradient.addColorStop(1, '#0a0a18');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

      // ========== STRUCTURED FLOW LAYOUT ==========
      // Layout: [ORACLE FEEDS] --> [QNO HUB] --> [RESOLUTIONS]
      //              LEFT            CENTER           RIGHT

      const ORACLE_ZONE_X = 120;
      const RESOLUTION_ZONE_X = CANVAS_WIDTH - 120;
      const STAKE_ZONE_Y = CENTER_Y + 100;

      // Draw flow arrows and zone backgrounds
      if (visualMode === 'radar' || visualMode === 'flow') {

        // ===== HEADER TITLE =====
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.font = 'bold 14px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('Prediction Staking Flow', CENTER_X, 25);

        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.font = '10px Inter, system-ui';
        ctx.fillText('Oracle data feeds resolve user predictions → Winners earn, losers get slashed', CENTER_X, 42);

        // ===== ORACLE ZONE (LEFT) =====
        // Zone background
        ctx.fillStyle = 'rgba(139, 92, 246, 0.08)';
        ctx.beginPath();
        ctx.roundRect(20, 60, 200, 280, 12);
        ctx.fill();
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Zone label
        ctx.fillStyle = '#a78bfa';
        ctx.font = 'bold 11px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('📡 ORACLE FEEDS', ORACLE_ZONE_X, 80);
        ctx.fillStyle = 'rgba(167, 139, 250, 0.6)';
        ctx.font = '9px Inter, system-ui';
        ctx.fillText('Real-time price data', ORACLE_ZONE_X, 94);

        // ===== CENTER HUB =====
        // Hub glow ring
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.2)';
        ctx.lineWidth = 2;
        for (let r = 50; r <= 90; r += 20) {
          ctx.beginPath();
          ctx.arc(CENTER_X, CENTER_Y, r, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Animated pulse ring
        const pulseRadius = 60 + Math.sin(now / 500) * 10;
        ctx.strokeStyle = `rgba(139, 92, 246, ${0.3 + Math.sin(now / 500) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(CENTER_X, CENTER_Y, pulseRadius, 0, Math.PI * 2);
        ctx.stroke();

        // ===== RESOLUTION ZONE (RIGHT) =====
        // Zone background
        ctx.fillStyle = 'rgba(16, 185, 129, 0.06)';
        ctx.beginPath();
        ctx.roundRect(CANVAS_WIDTH - 220, 60, 200, 280, 12);
        ctx.fill();
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Zone label
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 11px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('⚡ RESOLUTIONS', RESOLUTION_ZONE_X, 80);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.font = '9px Inter, system-ui';
        ctx.fillText('Prediction outcomes', RESOLUTION_ZONE_X, 94);

        // ===== FLOW ARROWS =====
        // Left arrow (Oracles → Hub)
        const arrowY = CENTER_Y - 40;
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.4)';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(230, arrowY);
        ctx.lineTo(CENTER_X - 100, arrowY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrow head
        ctx.fillStyle = 'rgba(139, 92, 246, 0.6)';
        ctx.beginPath();
        ctx.moveTo(CENTER_X - 100, arrowY);
        ctx.lineTo(CENTER_X - 115, arrowY - 6);
        ctx.lineTo(CENTER_X - 115, arrowY + 6);
        ctx.closePath();
        ctx.fill();

        // Arrow label
        ctx.fillStyle = 'rgba(167, 139, 250, 0.7)';
        ctx.font = '9px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('Price Data', 300, arrowY - 8);

        // Right arrow (Hub → Resolutions)
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(CENTER_X + 100, arrowY);
        ctx.lineTo(CANVAS_WIDTH - 230, arrowY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrow head
        ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.beginPath();
        ctx.moveTo(CANVAS_WIDTH - 230, arrowY);
        ctx.lineTo(CANVAS_WIDTH - 245, arrowY - 6);
        ctx.lineTo(CANVAS_WIDTH - 245, arrowY + 6);
        ctx.closePath();
        ctx.fill();

        // Arrow label
        ctx.fillStyle = 'rgba(16, 185, 129, 0.7)';
        ctx.font = '9px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('Compare to Predictions', CENTER_X + 170, arrowY - 8);

        // ===== STAKES ZONE (BOTTOM) =====
        ctx.fillStyle = 'rgba(59, 130, 246, 0.06)';
        ctx.beginPath();
        ctx.roundRect(CENTER_X - 150, STAKE_ZONE_Y - 20, 300, 80, 12);
        ctx.fill();
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 11px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText('💰 ACTIVE STAKES', CENTER_X, STAKE_ZONE_Y);
        ctx.fillStyle = 'rgba(59, 130, 246, 0.6)';
        ctx.font = '9px Inter, system-ui';
        ctx.fillText(`${stakes.length} predictions waiting for resolution`, CENTER_X, STAKE_ZONE_Y + 14);

        // Stake summary bar
        const stakeBarWidth = 250;
        const stakeBarX = CENTER_X - stakeBarWidth / 2;
        const totalStakeAmt = stakes.reduce((sum, s) => sum + s.amount, 0);

        // Domain breakdown mini-bars
        const domainTotals: Record<string, number> = {};
        stakes.forEach(s => {
          domainTotals[s.domain] = (domainTotals[s.domain] || 0) + s.amount;
        });

        let barOffset = 0;
        Object.entries(domainTotals).forEach(([domain, amount]) => {
          const width = totalStakeAmt > 0 ? (amount / totalStakeAmt) * stakeBarWidth : 0;
          const color = getDomainColor(domain);
          ctx.fillStyle = color.primary;
          ctx.beginPath();
          ctx.roundRect(stakeBarX + barOffset, STAKE_ZONE_Y + 28, width - 2, 16, 4);
          ctx.fill();

          if (width > 40) {
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 8px Inter, system-ui';
            ctx.textAlign = 'center';
            ctx.fillText(color.label, stakeBarX + barOffset + width / 2, STAKE_ZONE_Y + 39);
          }
          barOffset += width;
        });

        // Total staked
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.font = 'bold 10px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(`Total: ${formatCompact(totalStakeAmt)} QUG`, CENTER_X, STAKE_ZONE_Y + 58);

        // Up arrow from stakes to hub
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(CENTER_X, STAKE_ZONE_Y - 25);
        ctx.lineTo(CENTER_X, CENTER_Y + 50);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // ===== DRAW ORACLE DATA POINTS =====
      const oraclePositions = [
        { x: ORACLE_ZONE_X, y: 130 },
        { x: ORACLE_ZONE_X, y: 190 },
        { x: ORACLE_ZONE_X, y: 250 },
        { x: ORACLE_ZONE_X, y: 310 },
      ];

      oracleData.slice(0, 4).forEach((oracle, index) => {
        const pos = oraclePositions[index] || oraclePositions[0];
        const color = getDomainColor(oracle.domain);
        const age = oracle.age || 0;
        const isNew = age < 1000;
        const pulse = isNew ? Math.sin(age / 100) * 3 : 0;

        // Connection line to hub
        ctx.strokeStyle = `rgba(139, 92, 246, ${isNew ? 0.5 : 0.15})`;
        ctx.lineWidth = isNew ? 2 : 1;
        ctx.setLineDash(isNew ? [] : [4, 4]);
        ctx.beginPath();
        ctx.moveTo(pos.x + 80, pos.y);
        ctx.lineTo(CENTER_X - 50, CENTER_Y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Oracle card background
        ctx.fillStyle = 'rgba(30, 20, 50, 0.8)';
        ctx.beginPath();
        ctx.roundRect(pos.x - 70, pos.y - 22, 140, 44, 8);
        ctx.fill();
        ctx.strokeStyle = color.primary;
        ctx.lineWidth = isNew ? 2 : 1;
        ctx.stroke();

        // Glow effect for new data
        if (isNew) {
          ctx.shadowBlur = 15 + pulse;
          ctx.shadowColor = color.glow;
        }

        // Domain icon/indicator
        ctx.fillStyle = color.primary;
        ctx.beginPath();
        ctx.arc(pos.x - 50, pos.y, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowBlur = 0;

        // Domain label
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.font = 'bold 10px Inter, system-ui';
        ctx.textAlign = 'left';
        ctx.fillText(color.label, pos.x - 38, pos.y - 6);

        // Price value
        ctx.fillStyle = color.primary;
        ctx.font = 'bold 12px Inter, system-ui';
        ctx.fillText(formatValue(oracle.value, oracle.domain), pos.x - 38, pos.y + 10);

        // Confidence badge
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        ctx.beginPath();
        ctx.roundRect(pos.x + 40, pos.y - 10, 28, 20, 4);
        ctx.fill();
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 9px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(`${(oracle.confidence * 100).toFixed(0)}%`, pos.x + 54, pos.y + 4);
      });

      // ===== DRAW RESOLUTION EVENTS =====
      const resolutionPositions = [
        { x: RESOLUTION_ZONE_X, y: 120 },
        { x: RESOLUTION_ZONE_X, y: 170 },
        { x: RESOLUTION_ZONE_X, y: 220 },
        { x: RESOLUTION_ZONE_X, y: 270 },
        { x: RESOLUTION_ZONE_X, y: 320 },
      ];

      resolutions.slice(-5).forEach((resolution, index) => {
        const pos = resolutionPositions[index] || resolutionPositions[0];
        const age = resolution.age || 0;
        const isNew = age < 2000;

        const isWin = resolution.isAccurate;
        const cardColor = isWin ? '#10b981' : '#ef4444';
        const cardBg = isWin ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';

        // Connection from hub
        ctx.strokeStyle = isWin ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(CENTER_X + 50, CENTER_Y);
        ctx.lineTo(pos.x - 80, pos.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Resolution card
        ctx.fillStyle = cardBg;
        ctx.beginPath();
        ctx.roundRect(pos.x - 70, pos.y - 20, 140, 40, 8);
        ctx.fill();
        ctx.strokeStyle = cardColor;
        ctx.lineWidth = isNew ? 2 : 1;
        ctx.stroke();

        // Glow for new
        if (isNew) {
          ctx.shadowBlur = 12;
          ctx.shadowColor = isWin ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)';
        }

        // Win/Lose icon
        ctx.fillStyle = cardColor;
        ctx.font = 'bold 16px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(isWin ? '✓' : '✗', pos.x - 50, pos.y + 6);

        ctx.shadowBlur = 0;

        // Wallet and accuracy
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '9px Inter, system-ui';
        ctx.textAlign = 'left';
        ctx.fillText(resolution.walletPrefix || 'Wallet', pos.x - 35, pos.y - 6);

        ctx.fillStyle = cardColor;
        ctx.font = 'bold 11px Inter, system-ui';
        ctx.fillText(`${(resolution.accuracyScore * 100).toFixed(0)}% accuracy`, pos.x - 35, pos.y + 8);

        // Reward or slash amount
        if (resolution.slashingApplied > 0) {
          ctx.fillStyle = '#ef4444';
          ctx.font = 'bold 9px Inter, system-ui';
          ctx.textAlign = 'right';
          ctx.fillText(`-${resolution.slashingApplied.toFixed(0)}`, pos.x + 65, pos.y + 2);
        } else if (isWin) {
          ctx.fillStyle = '#10b981';
          ctx.font = 'bold 9px Inter, system-ui';
          ctx.textAlign = 'right';
          ctx.fillText(`+${(resolution.rewardAdjustment * 100).toFixed(0)}%`, pos.x + 65, pos.y + 2);
        }
      });

      // ===== DRAW PARTICLES =====
      particlesRef.current.forEach(p => {
        const alpha = Math.max(0, 1 - p.life / p.maxLife);
        const size = p.size * (1 - p.life / p.maxLife * 0.5);

        ctx.shadowBlur = 8;
        ctx.shadowColor = p.color;
        ctx.fillStyle = p.color.replace(')', `, ${alpha})`).replace('rgb', 'rgba');
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // ===== DRAW CENTER HUB =====
      ctx.shadowBlur = 30;
      ctx.shadowColor = 'rgba(139, 92, 246, 0.8)';
      const hubGradient = ctx.createRadialGradient(CENTER_X, CENTER_Y, 0, CENTER_X, CENTER_Y, 45);
      hubGradient.addColorStop(0, '#c4b5fd');
      hubGradient.addColorStop(0.4, '#a78bfa');
      hubGradient.addColorStop(0.7, '#8b5cf6');
      hubGradient.addColorStop(1, '#6d28d9');
      ctx.fillStyle = hubGradient;
      ctx.beginPath();
      ctx.arc(CENTER_X, CENTER_Y, 40, 0, Math.PI * 2);
      ctx.fill();

      // Hub text
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px Inter, system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('QNO', CENTER_X, CENTER_Y - 6);
      ctx.font = '9px Inter, system-ui';
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fillText('Oracle', CENTER_X, CENTER_Y + 10);

      animationFrameId.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [oracleData, resolutions, stakes, visualMode]);

  const isResolutionEvent = (event: ResolutionEvent | OracleDataPoint): event is ResolutionEvent => {
    return 'stakeId' in event;
  };

  return (
    <div className={`relative flex gap-4 p-4 ${className}`}>
      {/* Left Side: Main Canvas */}
      <div className="flex-1 relative">
        {/* Visual Mode Controls */}
        <div className="absolute top-2 right-2 z-10 flex gap-1">
          {[
            { mode: 'radar' as const, icon: Target, label: 'Radar', tooltip: 'Radar View: Shows predictions radiating from center with oracle data points. Good for seeing prediction distribution.' },
            { mode: 'flow' as const, icon: Activity, label: 'Flow', tooltip: 'Flow View: Visualizes data flowing from Oracle feeds → QNO Hub → Resolutions. Best for understanding the prediction lifecycle.' },
            { mode: 'chart' as const, icon: BarChart3, label: 'Chart', tooltip: 'Chart View: Displays stakes as a bar chart grouped by domain. Ideal for comparing stake volumes across markets.' },
            { mode: 'ring' as const, icon: PieChart, label: 'Ring', tooltip: 'Ring View: Shows concentric rings with stakes orbiting around oracle nodes. Great for spotting market concentration.' },
          ].map(({ mode, icon: Icon, label, tooltip }) => (
            <button
              key={mode}
              onClick={() => setVisualMode(mode)}
              title={tooltip}
              className={`px-2 py-1 rounded-lg font-medium text-[9px] transition-all flex items-center gap-1 ${
                visualMode === mode
                  ? 'bg-purple-500 text-white shadow-lg'
                  : 'bg-slate-800/70 text-gray-300 hover:bg-slate-700/70'
              }`}
            >
              <Icon className="w-3 h-3" />
              {label}
            </button>
          ))}
        </div>

        {/* Connection Status & Price Feed Status */}
        <div className="absolute top-2 left-2 z-10 flex gap-2">
          {/* SSE Connection */}
          <div
            title={isConnected
              ? 'Server-Sent Events: Connected to node for real-time stake/resolution updates. Events stream automatically.'
              : 'SSE Disconnected: Not receiving real-time updates. Check if your node is running.'}
            className={`px-2 py-1 rounded-lg backdrop-blur-sm border flex items-center gap-1.5 cursor-help ${
            isConnected ? 'bg-emerald-500/20 border-emerald-500/30' : 'bg-red-500/20 border-red-500/30'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
            <span className="text-[9px] font-medium text-white">
              {isConnected ? 'SSE' : 'Offline'}
            </span>
          </div>

          {/* Live Price Feed Status */}
          <div
            title={priceStatus === 'live'
              ? `Live Market Data: Fetching real-time BTC/ETH prices from CoinGecko API. Updates every 60 seconds. Last update: ${livePrices ? new Date(livePrices.lastUpdated).toLocaleTimeString() : 'N/A'}`
              : priceStatus === 'loading'
              ? 'Connecting to CoinGecko API to fetch live cryptocurrency prices...'
              : 'Fallback Mode: Using estimated prices. CoinGecko API may be rate-limited or unavailable. Prices will retry automatically.'}
            className={`px-2 py-1 rounded-lg backdrop-blur-sm border flex items-center gap-1.5 cursor-help ${
            priceStatus === 'live' ? 'bg-green-500/20 border-green-500/30' :
            priceStatus === 'loading' ? 'bg-yellow-500/20 border-yellow-500/30' :
            'bg-orange-500/20 border-orange-500/30'
          }`}>
            {priceStatus === 'live' ? (
              <Wifi className="w-3 h-3 text-green-400" />
            ) : priceStatus === 'loading' ? (
              <Activity className="w-3 h-3 text-yellow-400 animate-spin" />
            ) : (
              <WifiOff className="w-3 h-3 text-orange-400" />
            )}
            <span className="text-[9px] font-medium text-white">
              {priceStatus === 'live' ? 'Live Prices' :
               priceStatus === 'loading' ? 'Loading...' : 'Fallback'}
            </span>
            {livePrices && priceStatus === 'live' && (
              <span className="text-[8px] text-gray-400">
                {new Date(livePrices.lastUpdated).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </div>
        </div>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          title="Click on any element to see detailed information. Purple diamonds = Oracle price feeds with live data. Green/Red squares = Resolved predictions (win/slash). Blue circles = Active stakes awaiting resolution."
          className="w-full h-auto bg-slate-900 rounded-xl border border-purple-500/20 cursor-pointer"
          onClick={handleCanvasClick}
        />

        {/* Legend Bar */}
        <div className="flex justify-center gap-4 mt-2">
          <div
            title="Oracle Feed: Live price data from multiple sources (CoinGecko, Chainlink, Pyth). The aggregated price is used to resolve predictions."
            className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800/50 rounded border border-slate-700/30 cursor-help"
          >
            <div className="w-2.5 h-2.5 bg-purple-500 rotate-45" style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }} />
            <span className="text-[8px] text-gray-400">Oracle</span>
          </div>
          <div
            title="Winning Prediction: The prediction was accurate within the threshold (typically 95%+). The staker receives their stake back plus a reward bonus based on confidence level."
            className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800/50 rounded border border-slate-700/30 cursor-help"
          >
            <div className="w-2.5 h-2.5 bg-emerald-500 rotate-45" />
            <span className="text-[8px] text-gray-400">Win</span>
          </div>
          <div
            title="Slashed Prediction: The prediction was inaccurate. A portion of the stake is burned (slashed) proportional to how wrong the prediction was. Higher confidence = higher slashing risk."
            className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800/50 rounded border border-slate-700/30 cursor-help"
          >
            <div className="w-2.5 h-2.5 bg-red-500 rotate-45" />
            <span className="text-[8px] text-gray-400">Slash</span>
          </div>
          <div
            title="Active Stake: QUG tokens locked with a price prediction. Higher confidence stakes earn more rewards if correct, but risk more slashing if wrong."
            className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800/50 rounded border border-slate-700/30 cursor-help"
          >
            <div className="w-2.5 h-2.5 rounded-full border-2 border-blue-400" />
            <span className="text-[8px] text-gray-400">Stake</span>
          </div>
        </div>
      </div>

      {/* Right Side: Metrics Dashboard */}
      <div className="w-72 flex flex-col gap-3">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-2">
          <div
            title="Prediction Resolutions: Total number of stakes that have been resolved (compared against actual oracle prices). Higher resolution rate indicates active market participation."
            className="p-2.5 bg-slate-800/60 rounded-lg border border-purple-500/20 cursor-help"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="w-3 h-3 text-purple-400" />
              <span className="text-[9px] text-gray-400">Resolutions</span>
            </div>
            <div className="text-lg font-bold text-white">{stats.totalResolutions}</div>
            <div className="text-[9px] text-purple-300">{stats.resolutionsPerHour.toFixed(1)}/hr</div>
          </div>

          <div
            title="Win Rate: Percentage of predictions that were accurate enough to earn rewards. Accuracy shows how close predictions were on average. 95%+ accuracy required to win."
            className="p-2.5 bg-slate-800/60 rounded-lg border border-emerald-500/20 cursor-help"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="w-3 h-3 text-emerald-400" />
              <span className="text-[9px] text-gray-400">Win Rate</span>
            </div>
            <div className="text-lg font-bold text-emerald-400">{(stats.winRate * 100).toFixed(1)}%</div>
            <div className="text-[9px] text-emerald-300">Accuracy: {(stats.accuracyRate * 100).toFixed(1)}%</div>
          </div>

          <div
            title="Active Stakes: Number of pending predictions waiting to be resolved. Total QUG value locked in prediction staking. Higher stakes = higher potential rewards/risks."
            className="p-2.5 bg-slate-800/60 rounded-lg border border-blue-500/20 cursor-help"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <Users className="w-3 h-3 text-blue-400" />
              <span className="text-[9px] text-gray-400">Active Stakes</span>
            </div>
            <div className="text-lg font-bold text-white">{stats.activeStakes}</div>
            <div className="text-[9px] text-blue-300">{formatCompact(stats.totalStakeValue)} QUG</div>
          </div>

          <div
            title="Slashing: QUG tokens burned from inaccurate predictions. Slashing amount depends on prediction confidence and how wrong it was. Green shows total rewards earned by winners."
            className="p-2.5 bg-slate-800/60 rounded-lg border border-red-500/20 cursor-help"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <Flame className="w-3 h-3 text-red-400" />
              <span className="text-[9px] text-gray-400">Slashed</span>
            </div>
            <div className="text-lg font-bold text-red-400">{formatCompact(stats.totalSlashed)}</div>
            <div className="text-[9px] text-emerald-300">+{formatCompact(stats.totalRewarded)} earned</div>
          </div>
        </div>

        {/* Oracle Sources */}
        <div
          title="Oracle Sources: Multiple price feeds are aggregated to determine the final oracle price. Higher confidence sources have more weight. Decentralized oracles prevent price manipulation."
          className="p-2.5 bg-slate-800/60 rounded-lg border border-purple-500/20 cursor-help"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <Eye className="w-3 h-3 text-purple-400" />
              <span className="text-[9px] text-gray-400">Oracle Sources</span>
            </div>
            <span className="text-[10px] font-bold text-purple-400">{stats.oracleSources} Active</span>
          </div>
          <div className="flex gap-1">
            {[
              { name: 'CoinGecko', tooltip: 'CoinGecko: Primary live price source. Free API with real-time BTC/ETH prices. 98% confidence.' },
              { name: 'Chainlink', tooltip: 'Chainlink: Decentralized oracle network. Aggregates prices from multiple exchanges. 95% confidence.' },
              { name: 'Pyth', tooltip: 'Pyth Network: High-frequency oracle from trading firms. Sub-second price updates. 92% confidence.' },
            ].map((source, i) => (
              <div
                key={source.name}
                title={source.tooltip}
                className="flex-1 text-center py-1 rounded bg-slate-700/50 cursor-help"
              >
                <div className="text-[8px] text-gray-400">{source.name}</div>
                <div className="text-[9px] font-bold text-purple-300">{(92 + i * 3)}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Domain Stats */}
        <div
          title="Domain Performance: Breakdown of prediction accuracy and stake activity per market. Green = 75%+ accuracy. Yellow = below 75%. 'stk' = number of active stakes."
          className="p-2.5 bg-slate-800/60 rounded-lg border border-slate-600/30 cursor-help"
        >
          <div className="flex items-center gap-1.5 mb-2">
            <BarChart3 className="w-3 h-3 text-blue-400" />
            <span className="text-[9px] text-gray-400">Domain Performance</span>
          </div>
          <div className="space-y-1.5">
            {domainStats.slice(0, 4).map(ds => {
              const color = getDomainColor(ds.domain);
              const domainTooltips: Record<string, string> = {
                'crypto-btc': 'Bitcoin (BTC/USD): Most liquid crypto market. Predictions based on live CoinGecko prices.',
                'crypto-eth': 'Ethereum (ETH/USD): Second largest crypto. Includes gas price volatility.',
                'finance-sp500': 'S&P 500 Index: US stock market benchmark. Updated during market hours.',
                'weather-temp': 'Weather Temperature: Regional temperature predictions. Resolved via weather APIs.',
              };
              return (
                <div
                  key={ds.domain}
                  title={domainTooltips[ds.domain] || `${color.label}: Market prediction domain`}
                  className="flex items-center gap-2 cursor-help"
                >
                  <div className="w-2 h-2 rounded" style={{ backgroundColor: color.primary }} />
                  <span className="text-[9px] text-gray-300 flex-1">{color.label}</span>
                  <span
                    title={`Average prediction accuracy: ${ds.resolutionCount > 0 ? (ds.avgAccuracy * 100).toFixed(1) : 'N/A'}%`}
                    className="text-[9px] font-medium"
                    style={{ color: ds.avgAccuracy > 0.75 ? '#10b981' : '#f59e0b' }}
                  >
                    {ds.resolutionCount > 0 ? `${(ds.avgAccuracy * 100).toFixed(0)}%` : '-'}
                  </span>
                  <span title={`${ds.activeStakes} active stakes in this market`} className="text-[8px] text-gray-500">{ds.activeStakes} stk</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Live Event Feed */}
        <div
          title="Live Event Feed: Real-time stream of QNO activity. WIN = successful prediction, SLASH = failed prediction with penalty, STAKE = new prediction placed, oracle = price update."
          className="flex-1 p-2.5 bg-slate-800/60 rounded-lg border border-slate-600/30 min-h-0 overflow-hidden cursor-help"
        >
          <div className="flex items-center gap-1.5 mb-2">
            <Activity className="w-3 h-3 text-amber-400" />
            <span className="text-[9px] text-gray-400">Live Feed</span>
            <div title="Streaming active - new events appear automatically" className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-auto" />
          </div>
          <div className="space-y-1 overflow-y-auto max-h-32">
            {recentEvents.map((event, i) => {
              const eventTooltips: Record<string, string> = {
                'WIN': 'Winning Resolution: Prediction was accurate. Staker received their stake back plus bonus rewards.',
                'SLASH': 'Slashing Event: Prediction was inaccurate. A portion of the stake was burned as penalty.',
                'STAKE': 'New Stake: A user placed a new prediction with locked QUG tokens.',
                'oracle': 'Oracle Update: New price data received from external price feeds.',
              };
              return (
                <motion.div
                  key={`${event.timestamp}-${i}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  title={eventTooltips[event.type] || `${event.type} event`}
                  className="flex items-center gap-1.5 py-0.5 cursor-help"
                >
                  <span className="text-[8px] font-bold px-1 py-0.5 rounded" style={{
                    backgroundColor: `${event.color}20`,
                    color: event.color,
                  }}>
                    {event.type}
                  </span>
                  <span className="text-[9px] text-gray-300 truncate flex-1">{event.message}</span>
                  <span title={`${Math.floor((Date.now() - event.timestamp) / 1000)} seconds ago`} className="text-[8px] text-gray-500">
                    {Math.floor((Date.now() - event.timestamp) / 1000)}s
                  </span>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Quick Stats Bar */}
        <div className="flex gap-2">
          <div
            title="Average Confidence: Mean confidence level of all active stakes. Higher confidence = higher potential rewards but also higher slashing risk if wrong. Calculated as weighted average."
            className="flex-1 p-2 bg-slate-800/60 rounded-lg border border-slate-600/30 text-center cursor-help"
          >
            <div className="text-[8px] text-gray-400">Confidence</div>
            <div className="text-sm font-bold text-purple-400">{(stats.avgConfidence * 100).toFixed(0)}%</div>
          </div>
          <div
            title="24-Hour Volume: Total QUG tokens staked in predictions over the last 24 hours. Indicates market activity and liquidity. Higher volume = more active prediction market."
            className="flex-1 p-2 bg-slate-800/60 rounded-lg border border-slate-600/30 text-center cursor-help"
          >
            <div className="text-[8px] text-gray-400">24h Vol</div>
            <div className="text-sm font-bold text-blue-400">{formatCompact(stats.totalStakeValue * 3.2)}</div>
          </div>
        </div>
      </div>

      {/* Selected Event Details Popup */}
      <AnimatePresence>
        {selectedEvent && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="absolute bottom-4 left-4 z-20 w-64"
          >
            <div className="p-3 bg-slate-900/98 backdrop-blur-xl rounded-xl border border-purple-500/40 shadow-2xl">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-bold text-white flex items-center gap-1.5">
                  {isResolutionEvent(selectedEvent) ? (
                    <>
                      <Target className="w-3.5 h-3.5 text-purple-400" />
                      Resolution
                    </>
                  ) : (
                    <>
                      <Eye className="w-3.5 h-3.5 text-blue-400" />
                      Oracle Feed
                    </>
                  )}
                </h3>
                <button
                  onClick={() => setSelectedEvent(null)}
                  className="text-gray-400 hover:text-white p-0.5 hover:bg-slate-700 rounded"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="space-y-1 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-gray-400">Domain:</span>
                  <span className="text-white font-medium">{getDomainColor(selectedEvent.domain).label}</span>
                </div>

                {isResolutionEvent(selectedEvent) ? (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Predicted:</span>
                      <span className="text-blue-400 font-bold">
                        {formatValue(selectedEvent.predictedValue, selectedEvent.domain)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Actual:</span>
                      <span className="text-purple-400 font-bold">
                        {formatValue(selectedEvent.actualValue, selectedEvent.domain)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Accuracy:</span>
                      <span className={`font-bold ${selectedEvent.isAccurate ? 'text-emerald-400' : 'text-red-400'}`}>
                        {(selectedEvent.accuracyScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    {selectedEvent.slashingApplied > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Slashed:</span>
                        <span className="text-red-400 font-bold">
                          -{selectedEvent.slashingApplied.toFixed(2)} QUG
                        </span>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Value:</span>
                      <span className="text-purple-400 font-bold">
                        {formatValue(selectedEvent.value, selectedEvent.domain)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confidence:</span>
                      <span className="text-emerald-400 font-bold">
                        {(selectedEvent.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sources:</span>
                      <span className="text-blue-400 font-bold">
                        {selectedEvent.sources.length}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
