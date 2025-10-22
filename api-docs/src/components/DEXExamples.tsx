import { useState } from 'react';
import { motion } from 'framer-motion';
import { Network, TrendingUp, Code2, Zap, Shield, CheckCircle2, Copy, ChevronDown, ChevronRight } from 'lucide-react';

interface CodeExample {
  title: string;
  description: string;
  language: string;
  code: string;
}

function CodeBlock({ example }: { example: CodeExample }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(example.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      className="border border-quantum-purple/30 rounded-xl overflow-hidden bg-quantum-indigo/10 backdrop-blur-xl"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-quantum-purple/5 transition-all"
      >
        <div className="flex items-center gap-3">
          <Code2 className="w-5 h-5 text-quantum-cyan" />
          <div className="text-left">
            <h3 className="text-white font-bold">{example.title}</h3>
            <p className="text-sm text-gray-400">{example.description}</p>
          </div>
        </div>
        {expanded ? <ChevronDown className="w-5 h-5 text-gray-400" /> : <ChevronRight className="w-5 h-5 text-gray-400" />}
      </button>
      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="px-6 pb-6"
        >
          <div className="relative">
            <button
              onClick={copyToClipboard}
              className="absolute top-3 right-3 flex items-center gap-2 px-3 py-1 text-sm bg-quantum-purple/20 hover:bg-quantum-purple/30 rounded-lg text-white transition-all"
            >
              {copied ? <CheckCircle2 className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? 'Copied!' : 'Copy'}
            </button>
            <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
              {example.code}
            </pre>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default function DEXExamples() {
  const codeExamples: CodeExample[] = [
    {
      title: "Complete DEX Class - TypeScript",
      description: "Full-featured DEX implementation with swap, liquidity, and price queries",
      language: "typescript",
      code: `// Complete DEX Implementation for Quillon
import axios from 'axios';

interface SwapParams {
  fromToken: string;
  toToken: string;
  amountIn: number;
  minAmountOut: number;
  walletAddress: string;
}

interface LiquidityParams {
  token0: string;
  token1: string;
  amount0: number;
  amount1: number;
  walletAddress: string;
}

class QullionDEX {
  private apiUrl: string;
  private authHeader: string;

  constructor(apiUrl: string = 'http://localhost:8080') {
    this.apiUrl = apiUrl;
    this.authHeader = '';
  }

  // Set authentication for protected endpoints
  setAuth(walletAddress: string, signature: string, timestamp: number) {
    this.authHeader = JSON.stringify({
      address: walletAddress,
      signature: signature,
      timestamp: timestamp,
      scheme: 'Ed25519'
    });
  }

  // Execute token swap
  async swap(params: SwapParams): Promise<any> {
    const response = await axios.post(
      \`\${this.apiUrl}/api/v1/dex/swap\`,
      {
        from_token: params.fromToken,
        to_token: params.toToken,
        amount_in: params.amountIn,
        min_amount_out: params.minAmountOut,
        wallet_address: params.walletAddress
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': this.authHeader
        }
      }
    );
    return response.data;
  }

  // Add liquidity to pool
  async addLiquidity(params: LiquidityParams): Promise<any> {
    const response = await axios.post(
      \`\${this.apiUrl}/api/v1/dex/liquidity/add\`,
      {
        token0: params.token0,
        token1: params.token1,
        amount0: params.amount0,
        amount1: params.amount1,
        wallet_address: params.walletAddress
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'X-Wallet-Auth': this.authHeader
        }
      }
    );
    return response.data;
  }

  // Get all liquidity pools
  async getPools(): Promise<any> {
    const response = await axios.get(\`\${this.apiUrl}/api/v1/dex/pools\`);
    return response.data;
  }

  // Get specific pool info
  async getPool(token0: string, token1: string): Promise<any> {
    const response = await axios.get(
      \`\${this.apiUrl}/api/v1/dex/pools/\${token0}/\${token1}\`
    );
    return response.data;
  }

  // Calculate swap output
  async calculateSwapOutput(
    fromToken: string,
    toToken: string,
    amountIn: number
  ): Promise<number> {
    const pool = await this.getPool(fromToken, toToken);
    if (!pool.success) throw new Error('Pool not found');

    // Constant product formula: x * y = k
    const reserve0 = pool.data.reserve0;
    const reserve1 = pool.data.reserve1;
    const amountInWithFee = amountIn * 997; // 0.3% fee
    const numerator = amountInWithFee * reserve1;
    const denominator = reserve0 * 1000 + amountInWithFee;
    return numerator / denominator;
  }
}

// Usage Example
async function example() {
  const dex = new QullionDEX('http://localhost:8080');

  // Set authentication (sign timestamp with your wallet)
  const timestamp = Math.floor(Date.now() / 1000);
  dex.setAuth(
    'qnk470294cd0...',
    '0xabc123...', // Signature of "timestamp:1234567890"
    timestamp
  );

  // Execute swap: 1 QUG → QUGUSD
  const swapResult = await dex.swap({
    fromToken: 'QUG',
    toToken: 'QUGUSD',
    amountIn: 100_000_000, // 1 QUG (8 decimals)
    minAmountOut: 950_000_000, // Min 9.5 QUGUSD (5% slippage)
    walletAddress: 'qnk470294cd0...'
  });

  console.log('Swap successful:', swapResult);
  console.log('Received:', swapResult.data.amount_out / 100_000_000, 'QUGUSD');

  // Add liquidity
  const liquidityResult = await dex.addLiquidity({
    token0: 'QUG',
    token1: 'QUGUSD',
    amount0: 1000_000_000, // 10 QUG
    amount1: 1000_000_000, // 10 QUGUSD
    walletAddress: 'qnk470294cd0...'
  });

  console.log('Liquidity added:', liquidityResult);

  // Get all pools
  const pools = await dex.getPools();
  console.log('Available pools:', pools.data);
}

export default QullionDEX;`
    },
    {
      title: "React DEX Component",
      description: "Complete React component for token swapping with real-time price updates",
      language: "tsx",
      code: `import React, { useState, useEffect } from 'react';
import QullionDEX from './QullionDEX';

interface Token {
  symbol: string;
  name: string;
  balance: number;
  decimals: number;
}

const SwapComponent: React.FC = () => {
  const [dex] = useState(new QullionDEX('http://localhost:8080'));
  const [fromToken, setFromToken] = useState<Token>({
    symbol: 'QUG',
    name: 'Quillon',
    balance: 0,
    decimals: 8
  });
  const [toToken, setToToken] = useState<Token>({
    symbol: 'QUGUSD',
    name: 'Quillon USD',
    balance: 0,
    decimals: 8
  });
  const [amountIn, setAmountIn] = useState<string>('');
  const [amountOut, setAmountOut] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [priceImpact, setPriceImpact] = useState<number>(0);

  // Calculate output amount when input changes
  useEffect(() => {
    if (!amountIn || parseFloat(amountIn) === 0) {
      setAmountOut('');
      return;
    }

    const calculateOutput = async () => {
      try {
        const amountInBaseUnits = parseFloat(amountIn) * Math.pow(10, fromToken.decimals);
        const output = await dex.calculateSwapOutput(
          fromToken.symbol,
          toToken.symbol,
          amountInBaseUnits
        );
        setAmountOut((output / Math.pow(10, toToken.decimals)).toFixed(6));

        // Calculate price impact
        const expectedPrice = 1.0; // Assuming 1:1 for QUG/QUGUSD
        const actualPrice = output / amountInBaseUnits;
        const impact = ((expectedPrice - actualPrice) / expectedPrice) * 100;
        setPriceImpact(Math.abs(impact));
      } catch (error) {
        console.error('Price calculation failed:', error);
      }
    };

    const debounceTimer = setTimeout(calculateOutput, 500);
    return () => clearTimeout(debounceTimer);
  }, [amountIn, fromToken, toToken, dex]);

  const handleSwap = async () => {
    if (!amountIn || !amountOut) return;

    setLoading(true);
    try {
      const amountInBaseUnits = parseFloat(amountIn) * Math.pow(10, fromToken.decimals);
      const minAmountOut = parseFloat(amountOut) * Math.pow(10, toToken.decimals) * 0.995; // 0.5% slippage

      const result = await dex.swap({
        fromToken: fromToken.symbol,
        toToken: toToken.symbol,
        amountIn: amountInBaseUnits,
        minAmountOut: Math.floor(minAmountOut),
        walletAddress: 'qnk470294cd0...' // Get from wallet connection
      });

      if (result.success) {
        alert(\`Swap successful! Received \${result.data.amount_out / Math.pow(10, toToken.decimals)} \${toToken.symbol}\`);
        setAmountIn('');
        setAmountOut('');
      } else {
        alert(\`Swap failed: \${result.error}\`);
      }
    } catch (error: any) {
      alert(\`Error: \${error.message}\`);
    } finally {
      setLoading(false);
    }
  };

  const switchTokens = () => {
    const temp = fromToken;
    setFromToken(toToken);
    setToToken(temp);
    setAmountIn(amountOut);
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-gray-900 rounded-2xl border border-purple-500/30">
      <h2 className="text-2xl font-bold text-white mb-6">Swap Tokens</h2>

      {/* From Token */}
      <div className="mb-4 p-4 bg-gray-800 rounded-xl">
        <label className="text-gray-400 text-sm">From</label>
        <div className="flex items-center justify-between mt-2">
          <input
            type="number"
            value={amountIn}
            onChange={(e) => setAmountIn(e.target.value)}
            placeholder="0.0"
            className="bg-transparent text-2xl text-white outline-none w-full"
          />
          <div className="flex items-center gap-2 bg-gray-700 px-3 py-2 rounded-lg">
            <span className="text-white font-bold">{fromToken.symbol}</span>
          </div>
        </div>
        <p className="text-gray-500 text-sm mt-1">Balance: {fromToken.balance.toFixed(4)}</p>
      </div>

      {/* Switch Button */}
      <div className="flex justify-center -my-2 relative z-10">
        <button
          onClick={switchTokens}
          className="bg-gray-700 hover:bg-gray-600 p-2 rounded-lg transition-all"
        >
          ⇅
        </button>
      </div>

      {/* To Token */}
      <div className="mb-4 p-4 bg-gray-800 rounded-xl">
        <label className="text-gray-400 text-sm">To</label>
        <div className="flex items-center justify-between mt-2">
          <input
            type="text"
            value={amountOut}
            readOnly
            placeholder="0.0"
            className="bg-transparent text-2xl text-white outline-none w-full"
          />
          <div className="flex items-center gap-2 bg-gray-700 px-3 py-2 rounded-lg">
            <span className="text-white font-bold">{toToken.symbol}</span>
          </div>
        </div>
        <p className="text-gray-500 text-sm mt-1">Balance: {toToken.balance.toFixed(4)}</p>
      </div>

      {/* Price Impact */}
      {priceImpact > 0 && (
        <div className={\`mb-4 p-3 rounded-lg \${priceImpact > 5 ? 'bg-red-500/20 border border-red-500/50' : 'bg-yellow-500/20 border border-yellow-500/50'}\`}>
          <p className="text-sm">
            Price Impact: <span className="font-bold">{priceImpact.toFixed(2)}%</span>
            {priceImpact > 5 && ' ⚠️ High impact!'}
          </p>
        </div>
      )}

      {/* Swap Button */}
      <button
        onClick={handleSwap}
        disabled={loading || !amountIn || !amountOut}
        className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold rounded-xl transition-all"
      >
        {loading ? 'Swapping...' : 'Swap Tokens'}
      </button>
    </div>
  );
};

export default SwapComponent;`
    },
    {
      title: "Python DEX Client",
      description: "Python library for interacting with the Quillon DEX",
      language: "python",
      code: `import requests
import time
import hashlib
from typing import Dict, Any, Optional

class QullionDEX:
    """Quillon DEX Client - Python implementation"""

    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.auth_header = ""

    def set_auth(self, wallet_address: str, signature: str, timestamp: int):
        """Set authentication header for protected endpoints"""
        import json
        self.auth_header = json.dumps({
            "address": wallet_address,
            "signature": signature,
            "timestamp": timestamp,
            "scheme": "Ed25519"
        })

    def swap(
        self,
        from_token: str,
        to_token: str,
        amount_in: int,
        min_amount_out: int,
        wallet_address: str
    ) -> Dict[str, Any]:
        """Execute token swap"""
        response = requests.post(
            f"{self.api_url}/api/v1/dex/swap",
            json={
                "from_token": from_token,
                "to_token": to_token,
                "amount_in": amount_in,
                "min_amount_out": min_amount_out,
                "wallet_address": wallet_address
            },
            headers={
                "Content-Type": "application/json",
                "X-Wallet-Auth": self.auth_header
            }
        )
        return response.json()

    def add_liquidity(
        self,
        token0: str,
        token1: str,
        amount0: int,
        amount1: int,
        wallet_address: str
    ) -> Dict[str, Any]:
        """Add liquidity to pool"""
        response = requests.post(
            f"{self.api_url}/api/v1/dex/liquidity/add",
            json={
                "token0": token0,
                "token1": token1,
                "amount0": amount0,
                "amount1": amount1,
                "wallet_address": wallet_address
            },
            headers={
                "Content-Type": "application/json",
                "X-Wallet-Auth": self.auth_header
            }
        )
        return response.json()

    def get_pools(self) -> Dict[str, Any]:
        """Get all liquidity pools"""
        response = requests.get(f"{self.api_url}/api/v1/dex/pools")
        return response.json()

    def get_pool(self, token0: str, token1: str) -> Dict[str, Any]:
        """Get specific pool information"""
        response = requests.get(f"{self.api_url}/api/v1/dex/pools/{token0}/{token1}")
        return response.json()

    def calculate_swap_output(
        self,
        from_token: str,
        to_token: str,
        amount_in: int
    ) -> float:
        """Calculate expected output for a swap"""
        pool = self.get_pool(from_token, to_token)
        if not pool.get("success"):
            raise Exception("Pool not found")

        # Constant product formula: x * y = k
        # With 0.3% fee
        reserve0 = pool["data"]["reserve0"]
        reserve1 = pool["data"]["reserve1"]
        amount_in_with_fee = amount_in * 997  # 0.3% fee
        numerator = amount_in_with_fee * reserve1
        denominator = reserve0 * 1000 + amount_in_with_fee
        return numerator / denominator

# Usage Example
if __name__ == "__main__":
    # Initialize DEX client
    dex = QullionDEX("http://localhost:8080")

    # Set authentication
    timestamp = int(time.time())
    dex.set_auth(
        wallet_address="qnk470294cd0...",
        signature="0xabc123...",  # Sign "timestamp:1234567890"
        timestamp=timestamp
    )

    # Execute swap: 1 QUG → QUGUSD
    swap_result = dex.swap(
        from_token="QUG",
        to_token="QUGUSD",
        amount_in=100_000_000,  # 1 QUG (8 decimals)
        min_amount_out=950_000_000,  # Min 9.5 QUGUSD (5% slippage)
        wallet_address="qnk470294cd0..."
    )

    print("Swap successful:", swap_result)
    print(f"Received: {swap_result['data']['amount_out'] / 100_000_000} QUGUSD")

    # Add liquidity
    liquidity_result = dex.add_liquidity(
        token0="QUG",
        token1="QUGUSD",
        amount0=1000_000_000,  # 10 QUG
        amount1=1000_000_000,  # 10 QUGUSD
        wallet_address="qnk470294cd0..."
    )

    print("Liquidity added:", liquidity_result)

    # Get all pools
    pools = dex.get_pools()
    print("Available pools:")
    for pool in pools["data"]:
        print(f"  {pool['pair']}: TVL $" + f"{pool['tvl']:,.2f}, APR {pool['apr']:.2f}%")`
    },
    {
      title: "WebSocket Price Stream",
      description: "Real-time price updates using WebSocket subscriptions",
      language: "javascript",
      code: `// Real-time DEX Price Streaming
class DEXPriceStream {
  constructor(apiUrl = 'ws://localhost:8080') {
    this.ws = null;
    this.apiUrl = apiUrl;
    this.subscriptions = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(\`\${this.apiUrl}/api/v1/dex/stream\`);

      this.ws.onopen = () => {
        console.log('🔌 Connected to DEX price stream');
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };

      this.ws.onclose = () => {
        console.log('🔌 Disconnected from price stream');
        // Auto-reconnect after 5 seconds
        setTimeout(() => this.connect(), 5000);
      };
    });
  }

  handleMessage(data) {
    const { type, pair, price, volume, timestamp } = data;

    switch (type) {
      case 'price_update':
        this.notifySubscribers(pair, { price, volume, timestamp });
        break;
      case 'swap_executed':
        this.notifySubscribers(pair, {
          type: 'swap',
          amount: data.amount_in,
          output: data.amount_out,
          timestamp
        });
        break;
      case 'liquidity_added':
        this.notifySubscribers(pair, {
          type: 'liquidity',
          amount0: data.amount0,
          amount1: data.amount1,
          timestamp
        });
        break;
    }
  }

  subscribe(pair, callback) {
    if (!this.subscriptions.has(pair)) {
      this.subscriptions.set(pair, new Set());
      // Send subscription message to server
      this.ws.send(JSON.stringify({
        action: 'subscribe',
        pair: pair
      }));
    }
    this.subscriptions.get(pair).add(callback);

    // Return unsubscribe function
    return () => this.unsubscribe(pair, callback);
  }

  unsubscribe(pair, callback) {
    const callbacks = this.subscriptions.get(pair);
    if (callbacks) {
      callbacks.delete(callback);
      if (callbacks.size === 0) {
        this.subscriptions.delete(pair);
        this.ws.send(JSON.stringify({
          action: 'unsubscribe',
          pair: pair
        }));
      }
    }
  }

  notifySubscribers(pair, data) {
    const callbacks = this.subscriptions.get(pair);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Usage Example
async function example() {
  const stream = new DEXPriceStream('ws://localhost:8080');
  await stream.connect();

  // Subscribe to QUG/QUGUSD price updates
  const unsubscribe = stream.subscribe('QUG/QUGUSD', (data) => {
    if (data.type === 'swap') {
      console.log(\`💱 Swap executed: \${data.amount / 100_000_000} → \${data.output / 100_000_000}\`);
    } else {
      console.log(\`📊 Price: $\${data.price.toFixed(4)}, Volume: $\${data.volume.toFixed(2)}\`);
    }
  });

  // Subscribe to multiple pairs
  stream.subscribe('QUGUSD/USD', (data) => {
    console.log(\`💵 QUGUSD: $\${data.price.toFixed(6)}\`);
  });

  // Unsubscribe after 1 minute
  setTimeout(() => {
    unsubscribe();
    console.log('Unsubscribed from QUG/QUGUSD');
  }, 60000);
}

export default DEXPriceStream;`
    }
  ];

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">
          Build a Decentralized Exchange
        </h1>
        <p className="text-xl text-gray-300">
          Complete guide to building a high-performance DEX on Quillon with instant finality and 1M+ TPS
        </p>
      </motion.div>

      {/* Key Features */}
      <div className="grid md:grid-cols-3 gap-6">
        {[
          {
            icon: Zap,
            title: 'Instant Finality',
            description: '<50ms finality means no waiting for confirmations',
            color: 'from-quantum-cyan to-quantum-cyan/70'
          },
          {
            icon: TrendingUp,
            title: '1M+ TPS',
            description: 'Handle high-frequency trading with massive throughput',
            color: 'from-quantum-purple to-quantum-purple/70'
          },
          {
            icon: Shield,
            title: 'Post-Quantum Secure',
            description: 'Future-proof your DEX with quantum-resistant cryptography',
            color: 'from-quantum-pink to-quantum-pink/70'
          }
        ].map((feature, i) => (
          <motion.div
            key={feature.title}
            className="p-6 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <div className={`w-12 h-12 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-4`}>
              <feature.icon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
            <p className="text-gray-400">{feature.description}</p>
          </motion.div>
        ))}
      </div>

      {/* Architecture Overview */}
      <motion.div
        className="p-8 bg-gradient-to-br from-quantum-purple/20 to-quantum-cyan/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="flex items-center gap-3 mb-4">
          <Network className="w-7 h-7 text-quantum-purple" />
          <h2 className="text-2xl font-bold text-white">DEX Architecture</h2>
        </div>
        <div className="space-y-4 text-gray-300">
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">🏊 Liquidity Pools</h3>
            <p>Constant product (x * y = k) automated market maker with 0.3% trading fee distributed to liquidity providers</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">💱 Token Swaps</h3>
            <p>Execute swaps through liquidity pools with slippage protection and real-time price calculation</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">🔐 Wallet Authentication</h3>
            <p>All DEX operations require wallet signature authentication for security</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">📊 Real-time Updates</h3>
            <p>WebSocket streams provide instant price updates and trade notifications</p>
          </div>
        </div>
      </motion.div>

      {/* Code Examples */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
          <Code2 className="w-6 h-6 text-quantum-cyan" />
          Complete Code Examples
        </h2>
        {codeExamples.map((example, i) => (
          <CodeBlock key={i} example={example} />
        ))}
      </div>

      {/* Quick Start Guide */}
      <motion.div
        className="p-8 bg-quantum-green/10 backdrop-blur-xl border border-quantum-green/30 rounded-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
          <CheckCircle2 className="w-6 h-6 text-quantum-green" />
          Quick Start Checklist
        </h2>
        <div className="space-y-3">
          {[
            'Create wallet with POST /api/v1/wallets/create',
            'Get test tokens from faucet POST /api/v1/faucet',
            'Implement wallet authentication with signature',
            'Create liquidity pool POST /api/v1/dex/liquidity/add',
            'Execute swaps POST /api/v1/dex/swap',
            'Subscribe to real-time prices via WebSocket',
            'Build your UI with React/Vue/Angular components'
          ].map((step, i) => (
            <div key={i} className="flex items-start gap-3 p-3 bg-quantum-dark/30 rounded-lg">
              <CheckCircle2 className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <span className="text-gray-300">{step}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Best Practices */}
      <motion.div
        className="p-8 bg-quantum-purple/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">💡 Best Practices</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">Slippage Protection</h3>
            <p className="text-gray-400 text-sm">Always set min_amount_out to protect against price slippage (recommend 0.5-1%)</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">Price Impact</h3>
            <p className="text-gray-400 text-sm">Warn users when price impact exceeds 5% to prevent large losses</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">Balance Checks</h3>
            <p className="text-gray-400 text-sm">Always verify user balance before attempting swaps or liquidity operations</p>
          </div>
          <div className="p-4 bg-quantum-dark/30 rounded-lg">
            <h3 className="text-white font-bold mb-2">Error Handling</h3>
            <p className="text-gray-400 text-sm">Implement proper error handling for insufficient liquidity and failed transactions</p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
