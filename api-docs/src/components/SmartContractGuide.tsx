import { motion } from 'framer-motion';
import { Code2, Cpu, Zap, CheckCircle2, Terminal } from 'lucide-react';

export default function SmartContractGuide() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-quantum-cyan via-quantum-purple to-quantum-pink bg-clip-text text-transparent mb-4">
          Smart Contracts & Virtual Machine
        </h1>
        <p className="text-xl text-gray-300">
          Write smart contracts in Rust and deploy them instantly. No complex tooling, just compile and deploy.
        </p>
      </div>

      {/* Hero Feature */}
      <motion.div
        className="p-8 bg-gradient-to-br from-quantum-indigo/30 to-quantum-purple/20 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-4">
          <Cpu className="w-8 h-8 text-quantum-cyan" />
          <h2 className="text-2xl font-bold text-white">Q-VM: Rust-Native Smart Contracts</h2>
        </div>
        <p className="text-gray-300 mb-6">
          Our virtual machine executes compiled Rust WASM modules with native performance. 
          Write contracts in familiar Rust syntax with full type safety.
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { label: 'Native Performance', value: 'Near-native speed', icon: Zap },
            { label: 'Gas Efficient', value: '10x cheaper than EVM', icon: CheckCircle2 },
            { label: 'Type Safe', value: 'Rust compiler checks', icon: Code2 },
          ].map((stat) => (
            <div key={stat.label} className="p-4 bg-quantum-dark/50 rounded-xl border border-quantum-purple/20">
              <stat.icon className="w-6 h-6 text-quantum-cyan mb-2" />
              <div className="text-sm text-gray-400">{stat.label}</div>
              <div className="text-lg font-bold text-white">{stat.value}</div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Simple Token Contract */}
      <motion.div
        className="p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center gap-3 mb-4">
          <Code2 className="w-7 h-7 text-quantum-purple" />
          <h2 className="text-2xl font-bold text-white">Simple Token Contract</h2>
        </div>
        <p className="text-gray-300 mb-4">
          Create a fungible token in less than 50 lines of Rust:
        </p>
        <pre className="p-4 bg-quantum-dark/50 rounded-xl text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`use q_vm::prelude::*;

#[contract]
pub struct Token {
    balances: HashMap<Address, u64>,
    total_supply: u64,
}

#[contract_impl]
impl Token {
    pub fn new(initial_supply: u64) -> Self {
        let mut token = Token {
            balances: HashMap::new(),
            total_supply: initial_supply,
        };
        token.balances.insert(msg::sender(), initial_supply);
        token
    }

    pub fn transfer(&mut self, to: Address, amount: u64) -> Result<()> {
        let sender = msg::sender();
        let sender_balance = self.balances.get(&sender).copied().unwrap_or(0);
        
        require!(sender_balance >= amount, "Insufficient balance");
        
        *self.balances.entry(sender).or_insert(0) -= amount;
        *self.balances.entry(to).or_insert(0) += amount;
        
        emit!(Transfer { from: sender, to, amount });
        Ok(())
    }

    pub fn balance_of(&self, account: Address) -> u64 {
        self.balances.get(&account).copied().unwrap_or(0)
    }
}`}
        </pre>
      </motion.div>

      {/* Deploy Contract */}
      <motion.div
        className="p-8 bg-quantum-purple/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center gap-3 mb-4">
          <Terminal className="w-7 h-7 text-quantum-cyan" />
          <h2 className="text-2xl font-bold text-white">Deploy & Interact</h2>
        </div>
        <div className="space-y-4">
          <div>
            <h4 className="text-white font-bold mb-2">1. Compile Contract</h4>
            <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-green border border-quantum-purple/20 overflow-x-auto">
{`cargo build --target wasm32-unknown-unknown --release
wasm-opt -Oz -o token_optimized.wasm target/wasm32-unknown-unknown/release/token.wasm`}
            </pre>
          </div>
          <div>
            <h4 className="text-white font-bold mb-2">2. Deploy to Q-NarwhalKnight</h4>
            <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`curl -X POST http://localhost:8080/api/v1/contracts/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "wasm": "<base64_encoded_wasm>",
    "constructor_args": {"initial_supply": 1000000},
    "private_key": "..."
  }'

// Response: { "contract_address": "qnkc..." }`}
            </pre>
          </div>
          <div>
            <h4 className="text-white font-bold mb-2">3. Call Contract Methods</h4>
            <pre className="p-4 bg-quantum-dark/50 rounded-lg text-sm text-quantum-pink border border-quantum-purple/20 overflow-x-auto">
{`// Transfer tokens
curl -X POST http://localhost:8080/api/v1/contracts/call \
  -H "Content-Type: application/json" \
  -d '{
    "contract": "qnkc...",
    "method": "transfer",
    "args": {"to": "qnk...", "amount": 100},
    "private_key": "..."
  }'

// Check balance
curl http://localhost:8080/api/v1/contracts/query/qnkc.../balance_of?account=qnk...`}
            </pre>
          </div>
        </div>
      </motion.div>

      {/* Advanced Features */}
      <motion.div
        className="p-8 bg-quantum-cyan/10 backdrop-blur-xl border border-quantum-cyan/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">Advanced Features</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-bold text-quantum-cyan mb-3">Cross-Contract Calls</h3>
            <pre className="p-3 bg-quantum-dark/50 rounded-lg text-xs text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`pub fn swap(&mut self, token_address: Address) {
    let token = Token::at(token_address);
    token.transfer(msg::sender(), 100)?;
}`}
            </pre>
          </div>
          <div>
            <h3 className="text-lg font-bold text-quantum-cyan mb-3">Events & Logging</h3>
            <pre className="p-3 bg-quantum-dark/50 rounded-lg text-xs text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`#[event]
pub struct Transfer {
    from: Address,
    to: Address,
    amount: u64,
}

emit!(Transfer { from, to, amount });`}
            </pre>
          </div>
          <div>
            <h3 className="text-lg font-bold text-quantum-cyan mb-3">State Persistence</h3>
            <pre className="p-3 bg-quantum-dark/50 rounded-lg text-xs text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`// Automatic state serialization
#[contract]
pub struct DEX {
    pairs: HashMap<(Address, Address), Pool>,
    fees: u64,
}`}
            </pre>
          </div>
          <div>
            <h3 className="text-lg font-bold text-quantum-cyan mb-3">Gas Metering</h3>
            <pre className="p-3 bg-quantum-dark/50 rounded-lg text-xs text-quantum-cyan border border-quantum-purple/20 overflow-x-auto">
{`// Automatic gas tracking
let gas_used = contract.call(
    "transfer",
    args,
    GasLimit::Max(100000)
)?;`}
            </pre>
          </div>
        </div>
      </motion.div>

      {/* Why Q-VM */}
      <motion.div
        className="p-8 bg-quantum-indigo/20 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-2xl font-bold text-white mb-6">Why Q-VM is Better</h2>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            { title: 'Rust Native', desc: 'Use the most loved programming language, not a custom DSL' },
            { title: 'Type Safety', desc: 'Catch bugs at compile-time, not in production' },
            { title: 'WASM Performance', desc: 'Near-native execution speed, 10x faster than interpreted VMs' },
            { title: 'Small Binary Size', desc: 'Optimized WASM contracts are <10KB' },
            { title: 'No Solidity Quirks', desc: 'No reentrancy, no integer overflow, no undefined behavior' },
            { title: 'Future-Proof', desc: 'WASM is a web standard with massive ecosystem support' },
          ].map((benefit) => (
            <div key={benefit.title} className="flex items-start gap-3 p-4 bg-quantum-dark/30 rounded-xl">
              <CheckCircle2 className="w-5 h-5 text-quantum-green flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-bold text-white mb-1">{benefit.title}</h4>
                <p className="text-sm text-gray-400">{benefit.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Example: DEX Contract */}
      <motion.div
        className="p-8 bg-quantum-purple/10 backdrop-blur-xl border border-quantum-purple/30 rounded-2xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">Complete Example: Automated Market Maker</h2>
        <p className="text-gray-300 mb-4">
          Build a full DEX with liquidity pools in less than 100 lines:
        </p>
        <pre className="p-4 bg-quantum-dark/50 rounded-xl text-sm text-quantum-cyan border border-quantum-purple/20 overflow-x-auto max-h-96">
{`#[contract]
pub struct AMM {
    reserves: HashMap<(Address, Address), (u64, u64)>,
    lp_tokens: HashMap<(Address, Address), HashMap<Address, u64>>,
}

#[contract_impl]
impl AMM {
    pub fn add_liquidity(
        &mut self,
        token_a: Address,
        token_b: Address,
        amount_a: u64,
        amount_b: u64,
    ) -> Result<u64> {
        let pair = self.sort_pair(token_a, token_b);
        let (reserve_a, reserve_b) = self.reserves.entry(pair).or_insert((0, 0));
        
        // Transfer tokens from sender
        Token::at(token_a).transfer_from(msg::sender(), contract::address(), amount_a)?;
        Token::at(token_b).transfer_from(msg::sender(), contract::address(), amount_b)?;
        
        // Calculate LP tokens to mint
        let lp_amount = if *reserve_a == 0 {
            (amount_a * amount_b).sqrt()
        } else {
            min(
                amount_a * self.total_lp_supply(pair) / *reserve_a,
                amount_b * self.total_lp_supply(pair) / *reserve_b,
            )
        };
        
        // Mint LP tokens
        *self.lp_tokens.entry(pair).or_default().entry(msg::sender()).or_insert(0) += lp_amount;
        
        // Update reserves
        *reserve_a += amount_a;
        *reserve_b += amount_b;
        
        emit!(AddLiquidity { pair, amount_a, amount_b, lp_amount });
        Ok(lp_amount)
    }
    
    pub fn swap(
        &mut self,
        token_in: Address,
        token_out: Address,
        amount_in: u64,
        min_amount_out: u64,
    ) -> Result<u64> {
        let pair = self.sort_pair(token_in, token_out);
        let (reserve_in, reserve_out) = self.get_reserves(pair, token_in, token_out);
        
        // Calculate output with 0.3% fee
        let amount_out = self.get_amount_out(amount_in, reserve_in, reserve_out);
        require!(amount_out >= min_amount_out, "Slippage exceeded");
        
        // Execute swap
        Token::at(token_in).transfer_from(msg::sender(), contract::address(), amount_in)?;
        Token::at(token_out).transfer(msg::sender(), amount_out)?;
        
        // Update reserves
        self.update_reserves(pair, token_in, token_out, amount_in, amount_out);
        
        emit!(Swap { token_in, token_out, amount_in, amount_out });
        Ok(amount_out)
    }
    
    fn get_amount_out(&self, amount_in: u64, reserve_in: u64, reserve_out: u64) -> u64 {
        let amount_in_with_fee = amount_in * 997;
        let numerator = amount_in_with_fee * reserve_out;
        let denominator = reserve_in * 1000 + amount_in_with_fee;
        numerator / denominator
    }
}`}
        </pre>
      </motion.div>
    </div>
  );
}
