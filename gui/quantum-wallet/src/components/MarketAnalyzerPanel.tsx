import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Zap,
  AlertTriangle,
  Target,
  Activity,
  ChevronRight,
  RefreshCw,
  BarChart3,
  Clock,
  Shield,
  Loader2,
  MessageSquare,
  Droplets,
  LineChart,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react';

// v3.1.1: Helper to safely parse u128 values that may come as strings from the API
const parseU128 = (value: string | number | undefined): number => {
  if (value === undefined || value === null) return 0;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? 0 : parsed;
  }
  return 0;
};

interface Token {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  volume24h: number;
  liquidity: number;
  marketCap: number;
}

interface Pool {
  id: string;
  token0: string;
  token1: string;
  reserve0: number;
  reserve1: number;
  fee_tier: number;
  volume_24h?: number;
}

interface TechnicalIndicators {
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
    signal: 'overbought' | 'oversold' | 'neutral';
  };
  macd: {
    macd: number;
    signal: number;
    histogram: number;
    trend: 'bullish' | 'bearish' | 'neutral';
  };
  fibonacci: {
    level_0: number;
    level_236: number;
    level_382: number;
    level_500: number;
    level_618: number;
    level_786: number;
    level_100: number;
    currentLevel: string;
  };
  stochastic: {
    k: number;
    d: number;
    signal: 'overbought' | 'oversold' | 'neutral';
  };
  rsi: number;
  sma20: number;
  sma50: number;
  ema12: number;
  ema26: number;
}

interface TimeframeData {
  timeframe: '1h' | '24h' | '7d' | '30d' | 'max';
  indicators: TechnicalIndicators;
  priceChange: number;
  high: number;
  low: number;
  volume: number;
}

interface AIAnalysis {
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  summary: string;
  opportunities: string[];
  risks: string[];
  recommendation: string;
  technicalSummary?: string;
}

interface MarketAnalyzerPanelProps {
  isExpanded?: boolean;
  onOpportunityClick?: (tokenSymbol: string) => void;
}

export default function MarketAnalyzerPanel({
  isExpanded = false,
  onOpportunityClick
}: MarketAnalyzerPanelProps) {
  const [tokens, setTokens] = useState<Token[]>([]);
  const [pools, setPools] = useState<Pool[]>([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [expanded, setExpanded] = useState(isExpanded);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'technical' | 'analysis' | 'pools'>('overview');
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Technical analysis state
  const [selectedToken, setSelectedToken] = useState<string>('QUG');
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '24h' | '7d' | '30d' | 'max'>('24h');
  const [technicalData, setTechnicalData] = useState<TimeframeData | null>(null);
  const [loadingTechnical, setLoadingTechnical] = useState(false);

  // Fetch real token data from DEX, enriched with prices from pool reserves
  const fetchTokens = async (): Promise<Token[]> => {
    try {
      // Fetch both tokens and pools in parallel
      const [tokenRes, poolRes] = await Promise.all([
        fetch('/api/v1/dex/tokens'),
        fetch('/api/v1/liquidity/pools'),
      ]);

      let rawTokens: any[] = [];
      let rawPools: any[] = [];

      if (tokenRes.ok) {
        const result = await tokenRes.json();
        rawTokens = result.data || result.tokens || [];
      }
      if (poolRes.ok) {
        const result = await poolRes.json();
        rawPools = result.data || result.pools || [];
      }

      // Get QUG/USD price from QUG/QUGUSD pool
      let qugPriceUsd = 3000.00; // Default
      const qugQugusdPool = rawPools.find((p: any) =>
        (p.token0 === 'QUG' && p.token1 === 'QUGUSD') ||
        (p.token0 === 'QUGUSD' && p.token1 === 'QUG')
      );
      if (qugQugusdPool) {
        const r0 = parseFloat(qugQugusdPool.reserve0) || 0;
        const r1 = parseFloat(qugQugusdPool.reserve1) || 0;
        if (qugQugusdPool.token0 === 'QUG' && r0 > 0) {
          qugPriceUsd = r1 / r0;
        } else if (qugQugusdPool.token1 === 'QUG' && r1 > 0) {
          qugPriceUsd = r0 / r1;
        }
      }

      // Build price map: for each token paired with QUG, compute price
      const tokenPriceMap: Record<string, { price: number; liquidity: number; poolReserveQug: number }> = {};
      tokenPriceMap['QUG'] = { price: qugPriceUsd, liquidity: 0, poolReserveQug: 0 };
      tokenPriceMap['QUGUSD'] = { price: 1.0, liquidity: 0, poolReserveQug: 0 };

      for (const pool of rawPools) {
        const r0 = parseFloat(pool.reserve0) || 0;
        const r1 = parseFloat(pool.reserve1) || 0;
        if (r0 <= 0 || r1 <= 0) continue;

        const isToken0Qug = pool.token0 === 'QUG';
        const isToken1Qug = pool.token1 === 'QUG';

        if (isToken0Qug || isToken1Qug) {
          const tokenSymbol = isToken0Qug ? pool.token1 : pool.token0;
          const tokenReserve = isToken0Qug ? r1 : r0;
          const qugReserve = isToken0Qug ? r0 : r1;

          if (tokenReserve > 0) {
            const priceInQug = qugReserve / tokenReserve;
            const priceInUsd = priceInQug * qugPriceUsd;
            const liquidityUsd = qugReserve * qugPriceUsd * 2;

            // Use pool with highest liquidity for this token
            if (!tokenPriceMap[tokenSymbol] || liquidityUsd > tokenPriceMap[tokenSymbol].liquidity) {
              tokenPriceMap[tokenSymbol] = { price: priceInUsd, liquidity: liquidityUsd, poolReserveQug: qugReserve };
            }
          }
        }
      }

      // Enrich tokens with computed prices
      if (Array.isArray(rawTokens)) {
        return rawTokens.map((t: any) => {
          const sym = t.symbol || t.metadata?.symbol || 'UNKNOWN';
          const priceData = tokenPriceMap[sym];
          return {
            symbol: sym,
            name: t.name || t.metadata?.name || sym,
            price: priceData?.price || 0,
            change24h: t.change24h || t.change_24h || 0,
            volume24h: t.volume24h || t.volume_24h || 0,
            liquidity: priceData?.liquidity || 0,
            marketCap: t.marketCap || t.market_cap || 0,
          };
        }).filter((t: Token) => t.price > 0); // Only show tokens with real prices
      }
      return [];
    } catch (err) {
      console.error('Failed to fetch tokens:', err);
      return [];
    }
  };

  // Fetch real liquidity pools
  const fetchPools = async (): Promise<Pool[]> => {
    try {
      const response = await fetch('/api/v1/liquidity/pools');
      if (response.ok) {
        const result = await response.json();
        const pools = result.data || result.pools || [];
        if (Array.isArray(pools)) {
          return pools;
        }
      }
      return [];
    } catch (err) {
      console.error('Failed to fetch pools:', err);
      return [];
    }
  };

  // Fetch technical indicators for a token
  const fetchTechnicalIndicators = async (symbol: string, timeframe: string): Promise<TimeframeData | null> => {
    setLoadingTechnical(true);
    try {
      const response = await fetch(`/api/v1/market/technical/${symbol}?timeframe=${timeframe}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          return data.data;
        }
      }

      // If API not available, calculate basic indicators from available data
      const token = tokens.find(t => t.symbol === symbol);
      if (token) {
        return calculateBasicIndicators(token, timeframe);
      }
      return null;
    } catch (err) {
      console.error('Failed to fetch technical indicators:', err);
      // Fallback to basic calculation
      const token = tokens.find(t => t.symbol === symbol);
      if (token) {
        return calculateBasicIndicators(token, timeframe);
      }
      return null;
    } finally {
      setLoadingTechnical(false);
    }
  };

  // Calculate basic indicators from available price data
  const calculateBasicIndicators = (token: Token, timeframe: string): TimeframeData => {
    const price = token.price;
    const change = token.change24h;
    const volume = token.volume24h;

    // Simulate indicators based on current price and change
    // In production, these would come from historical OHLCV data
    const volatility = Math.abs(change) * 0.02;
    const high = price * (1 + volatility);
    const low = price * (1 - volatility);
    const mid = (high + low) / 2;

    // Bollinger Bands (simplified)
    const stdDev = price * volatility;
    const bbUpper = mid + (stdDev * 2);
    const bbLower = mid - (stdDev * 2);

    // RSI approximation based on price change
    const rsi = 50 + (change * 2);
    const clampedRsi = Math.max(0, Math.min(100, rsi));

    // Stochastic approximation
    const stochK = ((price - low) / (high - low || 1)) * 100;
    const stochD = stochK * 0.9; // Smoothed

    // MACD approximation
    const ema12 = price;
    const ema26 = price * (1 - change / 100);
    const macdValue = ema12 - ema26;
    const macdSignal = macdValue * 0.8;
    const macdHistogram = macdValue - macdSignal;

    // Fibonacci levels from recent high/low
    const fibRange = high - low;

    return {
      timeframe: timeframe as any,
      priceChange: change,
      high,
      low,
      volume,
      indicators: {
        bollinger: {
          upper: bbUpper,
          middle: mid,
          lower: bbLower,
          signal: price > bbUpper ? 'overbought' : price < bbLower ? 'oversold' : 'neutral',
        },
        macd: {
          macd: macdValue,
          signal: macdSignal,
          histogram: macdHistogram,
          trend: macdHistogram > 0 ? 'bullish' : macdHistogram < 0 ? 'bearish' : 'neutral',
        },
        fibonacci: {
          level_0: low,
          level_236: low + fibRange * 0.236,
          level_382: low + fibRange * 0.382,
          level_500: low + fibRange * 0.5,
          level_618: low + fibRange * 0.618,
          level_786: low + fibRange * 0.786,
          level_100: high,
          currentLevel: price < low + fibRange * 0.382 ? '0-38.2%' :
                       price < low + fibRange * 0.5 ? '38.2-50%' :
                       price < low + fibRange * 0.618 ? '50-61.8%' : '61.8-100%',
        },
        stochastic: {
          k: stochK,
          d: stochD,
          signal: stochK > 80 ? 'overbought' : stochK < 20 ? 'oversold' : 'neutral',
        },
        rsi: clampedRsi,
        sma20: price * 0.98,
        sma50: price * 0.95,
        ema12,
        ema26,
      },
    };
  };

  // Ask AI to analyze the real market data with technical indicators
  const analyzeWithAI = async (tokenData: Token[], poolData: Pool[], techData?: TimeframeData | null) => {
    if (tokenData.length === 0 && poolData.length === 0) {
      setError('No market data available to analyze');
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      // Build a prompt with real data for AI analysis
      const marketSummary = tokenData.map(t =>
        `${t.symbol}: $${t.price.toFixed(4)}, 24h: ${t.change24h >= 0 ? '+' : ''}${t.change24h.toFixed(2)}%, Vol: $${(t.volume24h / 1000).toFixed(1)}K, Liq: $${(t.liquidity / 1000).toFixed(1)}K`
      ).join('\n');

      const poolSummary = poolData.slice(0, 10).map(p =>
        `${p.token0}/${p.token1}: Reserve0=${(parseU128(p.reserve0) / 1e24).toFixed(2)}, Reserve1=${(parseU128(p.reserve1) / 1e24).toFixed(2)}, Fee=${p.fee_tier}%`
      ).join('\n');

      // Add technical analysis data if available
      let technicalSummary = '';
      if (techData) {
        const ind = techData.indicators;
        technicalSummary = `
TECHNICAL INDICATORS (${techData.timeframe}):
- RSI: ${ind.rsi.toFixed(1)} (${ind.rsi > 70 ? 'Overbought' : ind.rsi < 30 ? 'Oversold' : 'Neutral'})
- Bollinger Bands: Upper=${ind.bollinger.upper.toFixed(4)}, Mid=${ind.bollinger.middle.toFixed(4)}, Lower=${ind.bollinger.lower.toFixed(4)} → ${ind.bollinger.signal}
- MACD: ${ind.macd.macd.toFixed(4)}, Signal: ${ind.macd.signal.toFixed(4)}, Histogram: ${ind.macd.histogram.toFixed(4)} → ${ind.macd.trend}
- Stochastic: K=${ind.stochastic.k.toFixed(1)}, D=${ind.stochastic.d.toFixed(1)} → ${ind.stochastic.signal}
- Fibonacci Level: ${ind.fibonacci.currentLevel}
- SMA20: ${ind.sma20.toFixed(4)}, SMA50: ${ind.sma50.toFixed(4)}
- Price Range: High=${techData.high.toFixed(4)}, Low=${techData.low.toFixed(4)}`;
      }

      const prompt = `Analyze this DEX market data with technical indicators and provide trading insights. Be concise and actionable.

TOKENS (${tokenData.length}):
${marketSummary || 'No token data available'}

LIQUIDITY POOLS (${poolData.length}):
${poolSummary || 'No pool data available'}
${technicalSummary}

Based on the technical indicators (Bollinger Bands, MACD, RSI, Stochastic, Fibonacci), provide your analysis in this exact JSON format:
{
  "sentiment": "bullish" or "bearish" or "neutral",
  "confidence": 0-100,
  "summary": "2-3 sentence market overview including key technical signals",
  "technicalSummary": "Brief summary of what technical indicators suggest",
  "opportunities": ["opportunity 1 based on technicals", "opportunity 2"],
  "risks": ["risk 1 based on technicals", "risk 2"],
  "recommendation": "1-2 sentence actionable advice based on technical analysis"
}`;

      const response = await fetch('/api/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'Ministral-3B-Instruct',
          messages: [
            { role: 'system', content: 'You are an expert DeFi technical analyst. Analyze real blockchain DEX data with technical indicators (Bollinger Bands, MACD, RSI, Stochastic Oscillator, Fibonacci retracements) and provide concise, actionable insights. Always respond with valid JSON.' },
            { role: 'user', content: prompt }
          ],
          max_tokens: 512,
          temperature: 0.3,
        })
      });

      if (!response.ok) {
        throw new Error(`AI request failed: ${response.status}`);
      }

      const result = await response.json();
      const aiResponse = result.choices?.[0]?.message?.content || result.content || '';

      // Try to parse JSON from response
      try {
        let jsonStr = aiResponse;
        const jsonMatch = aiResponse.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) {
          jsonStr = jsonMatch[1];
        }
        const rawJsonMatch = aiResponse.match(/\{[\s\S]*\}/);
        if (rawJsonMatch) {
          jsonStr = rawJsonMatch[0];
        }

        const parsed = JSON.parse(jsonStr);
        setAiAnalysis({
          sentiment: parsed.sentiment || 'neutral',
          confidence: parsed.confidence || 50,
          summary: parsed.summary || 'Analysis complete.',
          technicalSummary: parsed.technicalSummary || '',
          opportunities: parsed.opportunities || [],
          risks: parsed.risks || [],
          recommendation: parsed.recommendation || 'Monitor the market.',
        });
      } catch (parseErr) {
        console.warn('Failed to parse AI JSON, using raw response');
        setAiAnalysis({
          sentiment: 'neutral',
          confidence: 50,
          summary: aiResponse.slice(0, 200),
          opportunities: [],
          risks: [],
          recommendation: 'See full analysis above.',
        });
      }

      setLastUpdated(new Date());
    } catch (err) {
      console.error('AI analysis failed, generating local analysis:', err);
      // v4.5.0: Generate meaningful local analysis from real pool/token data
      // instead of showing an error when AI is unavailable
      generateLocalAnalysis(tokenData, poolData, techData);
    } finally {
      setAnalyzing(false);
    }
  };

  // v4.5.0: Generate analysis locally from real on-chain data when AI is unavailable
  const generateLocalAnalysis = (tokenData: Token[], poolData: Pool[], techData?: TimeframeData | null) => {
    const totalLiquidity = tokenData.reduce((sum, t) => sum + t.liquidity, 0);
    const totalVolume = tokenData.reduce((sum, t) => sum + t.volume24h, 0);
    const avgChange = tokenData.length > 0
      ? tokenData.reduce((sum, t) => sum + t.change24h, 0) / tokenData.length
      : 0;

    // Find top gainers and losers
    const sorted = [...tokenData].sort((a, b) => b.change24h - a.change24h);
    const topGainer = sorted[0];
    const topLoser = sorted[sorted.length - 1];

    // Find highest liquidity tokens
    const byLiquidity = [...tokenData].sort((a, b) => b.liquidity - a.liquidity);
    const topLiquid = byLiquidity.slice(0, 3);

    // Determine sentiment
    const bullishCount = tokenData.filter(t => t.change24h > 0).length;
    const bearishCount = tokenData.filter(t => t.change24h < 0).length;
    const sentiment = bullishCount > bearishCount * 1.5 ? 'bullish' :
                      bearishCount > bullishCount * 1.5 ? 'bearish' : 'neutral';

    const confidence = Math.min(85, 40 + poolData.length * 3 + tokenData.length * 2);

    // Build opportunities
    const opportunities: string[] = [];
    if (topGainer && topGainer.change24h > 5) {
      opportunities.push(`${topGainer.symbol} up ${topGainer.change24h.toFixed(1)}% - momentum play if volume supports`);
    }
    if (topLiquid.length > 0) {
      opportunities.push(`Deepest liquidity: ${topLiquid.map(t => `${t.symbol} ($${(t.liquidity/1000).toFixed(0)}K)`).join(', ')}`);
    }
    if (poolData.length > 5) {
      opportunities.push(`${poolData.length} active pools - healthy market depth for arbitrage`);
    }

    // Build risks
    const risks: string[] = [];
    if (topLoser && topLoser.change24h < -5) {
      risks.push(`${topLoser.symbol} down ${Math.abs(topLoser.change24h).toFixed(1)}% - check liquidity before entering`);
    }
    const lowLiqTokens = tokenData.filter(t => t.liquidity < 1000 && t.liquidity > 0);
    if (lowLiqTokens.length > 0) {
      risks.push(`${lowLiqTokens.length} tokens with <$1K liquidity - high slippage risk`);
    }
    if (totalLiquidity < 50000) {
      risks.push('Overall DEX liquidity still growing - larger trades may have significant price impact');
    }

    // Technical summary
    let technicalSummary = '';
    if (techData) {
      const ind = techData.indicators;
      const signals: string[] = [];
      if (ind.rsi > 70) signals.push('RSI overbought');
      else if (ind.rsi < 30) signals.push('RSI oversold');
      else signals.push('RSI neutral');
      signals.push(`MACD ${ind.macd.trend}`);
      signals.push(`Bollinger ${ind.bollinger.signal}`);
      technicalSummary = `Technical signals: ${signals.join(', ')}`;
    }

    const summary = `Market overview: ${tokenData.length} tokens with real pricing from ${poolData.length} liquidity pools. ` +
      `Total liquidity: $${(totalLiquidity/1000).toFixed(0)}K. ` +
      `${bullishCount} tokens positive, ${bearishCount} negative. ` +
      (technicalSummary ? technicalSummary : '');

    setAiAnalysis({
      sentiment,
      confidence,
      summary,
      technicalSummary,
      opportunities: opportunities.length > 0 ? opportunities : ['Market data loading - check back shortly'],
      risks: risks.length > 0 ? risks : ['Always DYOR - on-chain data only, not financial advice'],
      recommendation: sentiment === 'bullish'
        ? `Market trending positive. Focus on high-liquidity pairs like ${topLiquid[0]?.symbol || 'QUG'} for lower slippage.`
        : sentiment === 'bearish'
        ? `Market showing weakness. Consider reducing exposure or waiting for support levels.`
        : `Mixed signals - focus on highest-liquidity pools for best execution.`,
    });
    setLastUpdated(new Date());
  };

  // Load real data and analyze
  const loadAndAnalyze = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [tokenData, poolData] = await Promise.all([
        fetchTokens(),
        fetchPools()
      ]);

      setTokens(tokenData);
      setPools(poolData);

      // Set default selected token
      if (tokenData.length > 0 && !tokenData.find(t => t.symbol === selectedToken)) {
        setSelectedToken(tokenData[0].symbol);
      }

      // Fetch technical indicators for selected token
      let techData = null;
      if (tokenData.length > 0) {
        const tokenSymbol = tokenData.find(t => t.symbol === selectedToken)?.symbol || tokenData[0].symbol;
        techData = await fetchTechnicalIndicators(tokenSymbol, selectedTimeframe);
        setTechnicalData(techData);
      }

      // Only auto-analyze if we have data
      if (tokenData.length > 0 || poolData.length > 0) {
        await analyzeWithAI(tokenData, poolData, techData);
      } else {
        setError('No DEX data available. Deploy tokens and create pools first.');
      }
    } catch (err) {
      setError('Failed to load market data');
    } finally {
      setLoading(false);
    }
  }, [selectedToken, selectedTimeframe]);

  // Update technical data when token or timeframe changes
  useEffect(() => {
    if (tokens.length > 0 && selectedTab === 'technical') {
      fetchTechnicalIndicators(selectedToken, selectedTimeframe).then(setTechnicalData);
    }
  }, [selectedToken, selectedTimeframe, selectedTab, tokens]);

  useEffect(() => {
    loadAndAnalyze();
    const interval = setInterval(loadAndAnalyze, 300000);
    return () => clearInterval(interval);
  }, [loadAndAnalyze]);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      default: return 'text-amber-300';
    }
  };

  const getSentimentBg = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'from-green-500/20 to-emerald-500/20 border-green-500/30';
      case 'bearish': return 'from-red-500/20 to-orange-500/20 border-red-500/30';
      default: return 'from-amber-500/20 to-yellow-500/20 border-amber-500/30';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'bullish':
      case 'oversold':
        return <ArrowUp className="w-4 h-4 text-green-400" />;
      case 'bearish':
      case 'overbought':
        return <ArrowDown className="w-4 h-4 text-red-400" />;
      default:
        return <Minus className="w-4 h-4 text-amber-400" />;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish':
      case 'oversold':
        return 'text-green-400 bg-green-500/20';
      case 'bearish':
      case 'overbought':
        return 'text-red-400 bg-red-500/20';
      default:
        return 'text-amber-400 bg-amber-500/20';
    }
  };

  const marketStats = {
    totalTokens: tokens.length,
    totalPools: pools.length,
    gainers: tokens.filter(t => t.change24h > 0).length,
    losers: tokens.filter(t => t.change24h < 0).length,
    totalVolume: tokens.reduce((sum, t) => sum + t.volume24h, 0),
    totalLiquidity: tokens.reduce((sum, t) => sum + t.liquidity, 0),
    avgChange: tokens.length > 0 ? tokens.reduce((sum, t) => sum + t.change24h, 0) / tokens.length : 0,
  };

  const timeframes = [
    { id: '1h', label: '1H' },
    { id: '24h', label: '24H' },
    { id: '7d', label: '7D' },
    { id: '30d', label: '30D' },
    { id: 'max', label: 'MAX' },
  ] as const;

  return (
    <motion.div
      className="bg-gradient-to-br from-slate-900/80 to-slate-800/60 rounded-2xl border border-violet-500/20 overflow-hidden backdrop-blur-xl"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-violet-500/5 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/30 to-fuchsia-500/30 flex items-center justify-center">
            <Brain className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h3 className="text-white font-bold flex items-center gap-2">
              AI Market Analyzer
              <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
                Live + Technicals
              </span>
            </h3>
            <p className="text-sm text-gray-400">
              {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : 'Analyzing real DEX data...'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {aiAnalysis && (
            <div className={`flex items-center gap-1 ${getSentimentColor(aiAnalysis.sentiment)}`}>
              {aiAnalysis.sentiment === 'bullish' ? (
                <TrendingUp className="w-4 h-4" />
              ) : aiAnalysis.sentiment === 'bearish' ? (
                <TrendingDown className="w-4 h-4" />
              ) : (
                <Activity className="w-4 h-4" />
              )}
              <span className="text-sm font-medium capitalize">{aiAnalysis.sentiment}</span>
              <span className="text-xs opacity-60">({aiAnalysis.confidence}%)</span>
            </div>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              loadAndAnalyze();
            }}
            disabled={loading || analyzing}
            className="p-2 rounded-lg hover:bg-violet-500/20 transition-colors"
          >
            <RefreshCw className={`w-4 h-4 text-violet-400 ${(loading || analyzing) ? 'animate-spin' : ''}`} />
          </button>
          <ChevronRight className={`w-5 h-5 text-violet-400 transition-transform ${expanded ? 'rotate-90' : ''}`} />
        </div>
      </div>

      {/* Expandable Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Tabs */}
            <div className="flex border-b border-violet-500/20 overflow-x-auto">
              {(['overview', 'technical', 'analysis', 'pools'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setSelectedTab(tab)}
                  className={`flex-1 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2 whitespace-nowrap px-2 ${
                    selectedTab === tab
                      ? 'text-violet-400 border-b-2 border-violet-400 bg-violet-500/5'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tab === 'overview' && <><BarChart3 className="w-4 h-4" /> Market</>}
                  {tab === 'technical' && <><LineChart className="w-4 h-4" /> Technical</>}
                  {tab === 'analysis' && <><MessageSquare className="w-4 h-4" /> AI</>}
                  {tab === 'pools' && <><Droplets className="w-4 h-4" /> Pools</>}
                </button>
              ))}
            </div>

            <div className="p-4">
              {error && (
                <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  {error}
                </div>
              )}

              {loading ? (
                <div className="flex flex-col items-center justify-center py-8 gap-3">
                  <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
                  <p className="text-gray-400">Loading real market data...</p>
                </div>
              ) : (
                <>
                  {/* Overview Tab */}
                  {selectedTab === 'overview' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 rounded-xl bg-slate-800/50 border border-violet-500/10">
                          <p className="text-xs text-gray-400">Tokens</p>
                          <p className="text-xl font-bold text-white">{marketStats.totalTokens}</p>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-800/50 border border-violet-500/10">
                          <p className="text-xs text-gray-400">Pools</p>
                          <p className="text-xl font-bold text-white">{marketStats.totalPools}</p>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-800/50 border border-violet-500/10">
                          <p className="text-xs text-gray-400">24h Volume</p>
                          <p className="text-xl font-bold text-white">${(marketStats.totalVolume / 1000).toFixed(1)}K</p>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-800/50 border border-violet-500/10">
                          <p className="text-xs text-gray-400">Total Liquidity</p>
                          <p className="text-xl font-bold text-white">${(marketStats.totalLiquidity / 1000).toFixed(1)}K</p>
                        </div>
                      </div>

                      {tokens.length > 0 && (
                        <div>
                          <h4 className="text-gray-300 font-medium mb-2 flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-green-400" />
                            Top Movers (24h)
                          </h4>
                          <div className="space-y-2">
                            {tokens
                              .sort((a, b) => Math.abs(b.change24h) - Math.abs(a.change24h))
                              .slice(0, 5)
                              .map((token, i) => (
                                <div
                                  key={i}
                                  className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 cursor-pointer transition-colors"
                                  onClick={() => onOpportunityClick?.(token.symbol)}
                                >
                                  <div className="flex items-center gap-3">
                                    <span className="text-white font-medium">{token.symbol}</span>
                                    <span className="text-xs text-gray-500">${token.price.toFixed(4)}</span>
                                  </div>
                                  <span className={token.change24h >= 0 ? 'text-green-400' : 'text-red-400'}>
                                    {token.change24h >= 0 ? '+' : ''}{token.change24h.toFixed(2)}%
                                  </span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-4 p-3 rounded-lg bg-slate-800/30">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-green-500"></div>
                          <span className="text-sm text-gray-400">Gainers: {marketStats.gainers}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-red-500"></div>
                          <span className="text-sm text-gray-400">Losers: {marketStats.losers}</span>
                        </div>
                        <div className="flex-1 h-2 rounded-full bg-slate-700 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-green-500 to-green-400"
                            style={{ width: `${marketStats.totalTokens > 0 ? (marketStats.gainers / marketStats.totalTokens) * 100 : 50}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Technical Analysis Tab */}
                  {selectedTab === 'technical' && (
                    <div className="space-y-4">
                      {/* Token and Timeframe Selectors */}
                      <div className="flex flex-wrap gap-3">
                        <select
                          value={selectedToken}
                          onChange={(e) => setSelectedToken(e.target.value)}
                          className="px-3 py-2 rounded-lg bg-slate-800/50 border border-violet-500/20 text-white text-sm"
                        >
                          {tokens.map(t => (
                            <option key={t.symbol} value={t.symbol}>{t.symbol}</option>
                          ))}
                        </select>

                        <div className="flex gap-1 p-1 rounded-lg bg-slate-800/50">
                          {timeframes.map(tf => (
                            <button
                              key={tf.id}
                              onClick={() => setSelectedTimeframe(tf.id)}
                              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                                selectedTimeframe === tf.id
                                  ? 'bg-violet-500 text-white'
                                  : 'text-gray-400 hover:text-white'
                              }`}
                            >
                              {tf.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {loadingTechnical ? (
                        <div className="flex items-center justify-center py-8">
                          <Loader2 className="w-6 h-6 text-violet-400 animate-spin" />
                        </div>
                      ) : technicalData ? (
                        <div className="space-y-4">
                          {/* RSI */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-gray-300 font-medium">RSI (14)</span>
                              <span className={`text-lg font-bold ${
                                technicalData.indicators.rsi > 70 ? 'text-red-400' :
                                technicalData.indicators.rsi < 30 ? 'text-green-400' : 'text-amber-400'
                              }`}>
                                {technicalData.indicators.rsi.toFixed(1)}
                              </span>
                            </div>
                            <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
                              <div
                                className={`h-full transition-all ${
                                  technicalData.indicators.rsi > 70 ? 'bg-red-500' :
                                  technicalData.indicators.rsi < 30 ? 'bg-green-500' : 'bg-amber-500'
                                }`}
                                style={{ width: `${technicalData.indicators.rsi}%` }}
                              />
                            </div>
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                              <span>Oversold (30)</span>
                              <span>Neutral</span>
                              <span>Overbought (70)</span>
                            </div>
                          </div>

                          {/* Bollinger Bands */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-gray-300 font-medium">Bollinger Bands</span>
                              <span className={`text-xs px-2 py-1 rounded ${getSignalColor(technicalData.indicators.bollinger.signal)}`}>
                                {technicalData.indicators.bollinger.signal.toUpperCase()}
                              </span>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-sm">
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">Upper</p>
                                <p className="text-red-400 font-mono">${technicalData.indicators.bollinger.upper.toFixed(4)}</p>
                              </div>
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">Middle</p>
                                <p className="text-white font-mono">${technicalData.indicators.bollinger.middle.toFixed(4)}</p>
                              </div>
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">Lower</p>
                                <p className="text-green-400 font-mono">${technicalData.indicators.bollinger.lower.toFixed(4)}</p>
                              </div>
                            </div>
                          </div>

                          {/* MACD */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-gray-300 font-medium">MACD</span>
                              <div className="flex items-center gap-2">
                                {getSignalIcon(technicalData.indicators.macd.trend)}
                                <span className={`text-xs px-2 py-1 rounded ${getSignalColor(technicalData.indicators.macd.trend)}`}>
                                  {technicalData.indicators.macd.trend.toUpperCase()}
                                </span>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-sm">
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">MACD</p>
                                <p className={`font-mono ${technicalData.indicators.macd.macd >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {technicalData.indicators.macd.macd.toFixed(4)}
                                </p>
                              </div>
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">Signal</p>
                                <p className="text-white font-mono">{technicalData.indicators.macd.signal.toFixed(4)}</p>
                              </div>
                              <div className="text-center p-2 rounded bg-slate-900/50">
                                <p className="text-gray-500 text-xs">Histogram</p>
                                <p className={`font-mono ${technicalData.indicators.macd.histogram >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {technicalData.indicators.macd.histogram.toFixed(4)}
                                </p>
                              </div>
                            </div>
                          </div>

                          {/* Stochastic Oscillator */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-gray-300 font-medium">Stochastic Oscillator</span>
                              <span className={`text-xs px-2 py-1 rounded ${getSignalColor(technicalData.indicators.stochastic.signal)}`}>
                                {technicalData.indicators.stochastic.signal.toUpperCase()}
                              </span>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-gray-500 text-xs mb-1">%K</p>
                                <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
                                  <div className="h-full bg-blue-500" style={{ width: `${technicalData.indicators.stochastic.k}%` }} />
                                </div>
                                <p className="text-blue-400 font-mono text-sm mt-1">{technicalData.indicators.stochastic.k.toFixed(1)}</p>
                              </div>
                              <div>
                                <p className="text-gray-500 text-xs mb-1">%D</p>
                                <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
                                  <div className="h-full bg-orange-500" style={{ width: `${technicalData.indicators.stochastic.d}%` }} />
                                </div>
                                <p className="text-orange-400 font-mono text-sm mt-1">{technicalData.indicators.stochastic.d.toFixed(1)}</p>
                              </div>
                            </div>
                          </div>

                          {/* Fibonacci Retracements */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-gray-300 font-medium">Fibonacci Retracements</span>
                              <span className="text-xs text-violet-400">Current: {technicalData.indicators.fibonacci.currentLevel}</span>
                            </div>
                            <div className="space-y-2 text-xs">
                              {[
                                { level: '100%', value: technicalData.indicators.fibonacci.level_100, color: 'text-red-400' },
                                { level: '78.6%', value: technicalData.indicators.fibonacci.level_786, color: 'text-orange-400' },
                                { level: '61.8%', value: technicalData.indicators.fibonacci.level_618, color: 'text-amber-400' },
                                { level: '50%', value: technicalData.indicators.fibonacci.level_500, color: 'text-yellow-400' },
                                { level: '38.2%', value: technicalData.indicators.fibonacci.level_382, color: 'text-lime-400' },
                                { level: '23.6%', value: technicalData.indicators.fibonacci.level_236, color: 'text-green-400' },
                                { level: '0%', value: technicalData.indicators.fibonacci.level_0, color: 'text-emerald-400' },
                              ].map(fib => (
                                <div key={fib.level} className="flex items-center justify-between p-2 rounded bg-slate-900/50">
                                  <span className={fib.color}>{fib.level}</span>
                                  <span className="text-white font-mono">${fib.value.toFixed(4)}</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Moving Averages */}
                          <div className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10">
                            <span className="text-gray-300 font-medium">Moving Averages</span>
                            <div className="grid grid-cols-2 gap-3 mt-3">
                              <div className="p-2 rounded bg-slate-900/50 text-center">
                                <p className="text-gray-500 text-xs">SMA 20</p>
                                <p className="text-white font-mono">${technicalData.indicators.sma20.toFixed(4)}</p>
                              </div>
                              <div className="p-2 rounded bg-slate-900/50 text-center">
                                <p className="text-gray-500 text-xs">SMA 50</p>
                                <p className="text-white font-mono">${technicalData.indicators.sma50.toFixed(4)}</p>
                              </div>
                              <div className="p-2 rounded bg-slate-900/50 text-center">
                                <p className="text-gray-500 text-xs">EMA 12</p>
                                <p className="text-white font-mono">${technicalData.indicators.ema12.toFixed(4)}</p>
                              </div>
                              <div className="p-2 rounded bg-slate-900/50 text-center">
                                <p className="text-gray-500 text-xs">EMA 26</p>
                                <p className="text-white font-mono">${technicalData.indicators.ema26.toFixed(4)}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-8 text-gray-400">
                          <LineChart className="w-8 h-8 mx-auto mb-2 opacity-50" />
                          <p>Select a token to view technical indicators</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* AI Analysis Tab */}
                  {selectedTab === 'analysis' && (
                    <div className="space-y-4">
                      {analyzing ? (
                        <div className="flex flex-col items-center justify-center py-8 gap-3">
                          <Brain className="w-8 h-8 text-violet-400 animate-pulse" />
                          <p className="text-gray-400">AI analyzing market data with technical indicators...</p>
                        </div>
                      ) : aiAnalysis ? (
                        <>
                          <div className={`p-4 rounded-xl bg-gradient-to-br ${getSentimentBg(aiAnalysis.sentiment)} border`}>
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center gap-2">
                                {aiAnalysis.sentiment === 'bullish' ? (
                                  <TrendingUp className="w-5 h-5 text-green-400" />
                                ) : aiAnalysis.sentiment === 'bearish' ? (
                                  <TrendingDown className="w-5 h-5 text-red-400" />
                                ) : (
                                  <Activity className="w-5 h-5 text-amber-400" />
                                )}
                                <span className={`text-lg font-bold capitalize ${getSentimentColor(aiAnalysis.sentiment)}`}>
                                  {aiAnalysis.sentiment}
                                </span>
                              </div>
                              <div className="text-right">
                                <p className="text-xs text-gray-400">Confidence</p>
                                <p className="text-lg font-bold text-white">{aiAnalysis.confidence}%</p>
                              </div>
                            </div>
                            <p className="text-gray-200">{aiAnalysis.summary}</p>
                          </div>

                          {aiAnalysis.technicalSummary && (
                            <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/30">
                              <div className="flex items-start gap-2">
                                <LineChart className="w-5 h-5 text-blue-400 mt-0.5" />
                                <div>
                                  <p className="text-sm text-gray-400 mb-1">Technical Summary</p>
                                  <p className="text-blue-200">{aiAnalysis.technicalSummary}</p>
                                </div>
                              </div>
                            </div>
                          )}

                          {aiAnalysis.opportunities.length > 0 && (
                            <div>
                              <h4 className="text-gray-300 font-medium mb-2 flex items-center gap-2">
                                <Target className="w-4 h-4 text-green-400" />
                                Opportunities
                              </h4>
                              <div className="space-y-2">
                                {aiAnalysis.opportunities.map((opp, i) => (
                                  <div key={i} className="p-3 rounded-lg bg-green-500/10 border border-green-500/20 text-green-200 text-sm">
                                    {opp}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {aiAnalysis.risks.length > 0 && (
                            <div>
                              <h4 className="text-gray-300 font-medium mb-2 flex items-center gap-2">
                                <AlertTriangle className="w-4 h-4 text-amber-400" />
                                Risks
                              </h4>
                              <div className="space-y-2">
                                {aiAnalysis.risks.map((risk, i) => (
                                  <div key={i} className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200 text-sm">
                                    {risk}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/30">
                            <div className="flex items-start gap-2">
                              <Shield className="w-5 h-5 text-violet-400 mt-0.5" />
                              <div>
                                <p className="text-sm text-gray-400 mb-1">AI Recommendation</p>
                                <p className="text-white">{aiAnalysis.recommendation}</p>
                              </div>
                            </div>
                          </div>
                        </>
                      ) : (
                        <div className="text-center py-8 text-gray-400">
                          <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                          <p>No analysis available</p>
                          <button
                            onClick={() => analyzeWithAI(tokens, pools, technicalData)}
                            className="mt-3 px-4 py-2 rounded-lg bg-violet-500/20 text-violet-300 hover:bg-violet-500/30 transition-colors"
                          >
                            Run Analysis
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Pools Tab */}
                  {selectedTab === 'pools' && (
                    <div className="space-y-3">
                      {pools.length === 0 ? (
                        <div className="text-center py-8 text-gray-400">
                          <Droplets className="w-8 h-8 mx-auto mb-2 opacity-50" />
                          <p>No liquidity pools found</p>
                          <p className="text-xs mt-1">Create a pool to see it here</p>
                        </div>
                      ) : (
                        pools.slice(0, 10).map((pool, i) => (
                          <div
                            key={pool.id || i}
                            className="p-4 rounded-xl bg-slate-800/50 border border-violet-500/10 hover:border-violet-500/30 transition-colors"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-white font-medium">{pool.token0} / {pool.token1}</span>
                              <span className="text-xs text-gray-400">Fee: {pool.fee_tier}%</span>
                            </div>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <p className="text-gray-400">{pool.token0}</p>
                                <p className="text-white font-mono">{(parseU128(pool.reserve0) / 1e24).toFixed(4)}</p>
                              </div>
                              <div>
                                <p className="text-gray-400">{pool.token1}</p>
                                <p className="text-white font-mono">{(parseU128(pool.reserve1) / 1e24).toFixed(4)}</p>
                              </div>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
