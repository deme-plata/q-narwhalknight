import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { UserPlus, Check, AlertCircle, Copy, CheckCheck, Wallet, Link as LinkIcon } from 'lucide-react'
import bountyApi, { type RegisterRequest } from '../services/api'
import { useWallet } from '../App'

export default function Register() {
  const { wallet, connect, connecting } = useWallet()
  const [walletAddress, setWalletAddress] = useState('')
  const [mainnetAddress, setMainnetAddress] = useState('')
  const [success, setSuccess] = useState(false)
  const [userId, setUserId] = useState('')
  const [copied, setCopied] = useState(false)

  // Auto-populate wallet address when connected via OAuth2
  useEffect(() => {
    if (wallet?.connected && wallet.address && !walletAddress) {
      setWalletAddress(wallet.address)
    }
  }, [wallet, walletAddress])

  const registerMutation = useMutation({
    mutationFn: (data: RegisterRequest) => bountyApi.register(data),
    onSuccess: (data) => {
      setSuccess(true)
      setUserId(data.user_id)
      // Store in localStorage for convenience
      localStorage.setItem('bounty_user_id', data.user_id)
      if (data.token) {
        localStorage.setItem('bounty_token', data.token)
      }
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!walletAddress.trim()) return

    registerMutation.mutate({
      testnet_address: walletAddress.trim(),
      mainnet_address: mainnetAddress.trim() || undefined,
    })
  }

  const copyUserId = () => {
    navigator.clipboard.writeText(userId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
          <span className="gradient-text">Join the Bounty</span>
        </h1>
        <p className="text-slate-400 text-lg">
          Register your wallet to start earning rewards for contributing to Q-NarwhalKnight
        </p>
      </div>

      {success && (
        <div className="glass-card p-6 mb-8 border-green-500/30">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
              <Check className="w-5 h-5 text-green-400" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-green-400 font-bold text-lg mb-2">Registration Successful!</h3>
              <p className="text-slate-300 text-sm mb-3">Your unique Bounty ID:</p>
              <div className="flex items-center gap-2">
                <code className="flex-1 bg-slate-900/80 px-4 py-2.5 rounded-lg text-green-400 font-mono text-sm break-all border border-green-500/20">
                  {userId}
                </code>
                <button
                  onClick={copyUserId}
                  className="p-2.5 bg-slate-900/80 rounded-lg border border-green-500/20 text-green-400 hover:bg-green-500/10 transition-colors"
                >
                  {copied ? <CheckCheck className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-slate-500 text-xs mt-3">
                Save this ID - you need it to track progress and claim rewards.
              </p>
            </div>
          </div>
        </div>
      )}

      {registerMutation.isError && (
        <div className="glass-card p-6 mb-8 border-red-500/30">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-red-400 font-semibold mb-1">Registration Failed</h3>
              <p className="text-slate-400 text-sm">
                {(registerMutation.error as Error)?.message || 'Please check your wallet address and try again.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Quick Connect Banner */}
      {!wallet?.connected && !success && (
        <div className="glass-card p-6 mb-8 border-purple-500/20">
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex items-center gap-3 flex-1">
              <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <LinkIcon className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Already using quillon.xyz?</h3>
                <p className="text-slate-400 text-xs">
                  Connect your wallet to auto-fill your address
                </p>
              </div>
            </div>
            <button
              onClick={connect}
              disabled={connecting}
              className="btn-shine flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 text-white text-sm font-semibold rounded-lg hover:from-purple-500 hover:to-blue-500 transition-all disabled:opacity-50 whitespace-nowrap"
            >
              {connecting ? (
                <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white" />
              ) : (
                <Wallet className="w-4 h-4" />
              )}
              {connecting ? 'Connecting...' : 'Connect Wallet'}
            </button>
          </div>
        </div>
      )}

      {/* Connected wallet badge */}
      {wallet?.connected && !success && (
        <div className="glass-card p-4 mb-8 border-green-500/20">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-green-300 text-sm font-medium">Wallet connected</span>
            <code className="text-xs text-slate-400 font-mono">
              {wallet.address.slice(0, 10)}...{wallet.address.slice(-6)}
            </code>
          </div>
        </div>
      )}

      <div className="glass-card p-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Wallet Address */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Wallet Address <span className="text-red-400">*</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={walletAddress}
                onChange={(e) => setWalletAddress(e.target.value)}
                placeholder="Your QNK wallet address (qnk... or hex)"
                className={`w-full px-4 py-3 bg-slate-900/50 border rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all ${
                  wallet?.connected ? 'border-green-500/30 pr-24' : 'border-white/10'
                }`}
                required
              />
              {wallet?.connected && walletAddress === wallet.address && (
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded">
                  Connected
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500 mt-1.5">
              {wallet?.connected
                ? 'Auto-filled from your connected Quillon wallet'
                : 'Your Q-NarwhalKnight wallet address from quillon.xyz'}
            </p>
          </div>

          {/* Mainnet Address (Optional) */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Mainnet Wallet Address <span className="text-slate-500">(optional)</span>
            </label>
            <input
              type="text"
              value={mainnetAddress}
              onChange={(e) => setMainnetAddress(e.target.value)}
              placeholder="For reward distribution (can be added later)"
              className="w-full px-4 py-3 bg-slate-900/50 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
            />
          </div>

          {/* Info */}
          <div className="bg-purple-500/5 border border-purple-500/15 rounded-xl p-5">
            <h4 className="text-purple-300 font-semibold mb-3 flex items-center gap-2 text-sm">
              <UserPlus className="w-4 h-4" />
              What happens after registration?
            </h4>
            <ul className="text-sm text-slate-400 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5">1.</span>
                You receive a unique Bounty ID to track your progress
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5">2.</span>
                Start contributing - run nodes, report bugs, create content
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5">3.</span>
                Points are scored across 5 weighted categories
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5">4.</span>
                Early participants earn up to 2x bonus multipliers
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-400 mt-0.5">5.</span>
                Rewards distributed based on your tier and total score
              </li>
            </ul>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={registerMutation.isPending || !walletAddress.trim()}
            className="btn-shine w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-500 hover:to-blue-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-purple-500/20"
          >
            {registerMutation.isPending ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                <span>Registering...</span>
              </>
            ) : (
              <>
                <UserPlus className="w-5 h-5" />
                <span>Register Now</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Tier Cards */}
      <div className="mt-12">
        <h3 className="text-2xl font-bold text-white mb-6 text-center">Reward Tiers</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { tier: 'Pioneer', pool: '20%', rank: 'Top 1%', vesting: '6mo', glow: 'tier-pioneer', badge: 'bg-yellow-500/15 border-yellow-500/30 text-yellow-300' },
            { tier: 'Contributor', pool: '30%', rank: 'Top 10%', vesting: '3mo', glow: 'tier-contributor', badge: 'bg-blue-500/15 border-blue-500/30 text-blue-300' },
            { tier: 'Participant', pool: '40%', rank: 'Top 50%', vesting: '1mo', glow: 'tier-participant', badge: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300' },
            { tier: 'Supporter', pool: '10%', rank: 'All', vesting: 'None', glow: 'tier-supporter', badge: 'bg-purple-500/15 border-purple-500/30 text-purple-300' },
          ].map((t) => (
            <div key={t.tier} className={`glass-card p-4 ${t.glow}`}>
              <div className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-bold border mb-3 ${t.badge}`}>
                {t.tier}
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-500">Pool</span>
                  <span className="text-white font-semibold">{t.pool}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Rank</span>
                  <span className="text-white font-semibold">{t.rank}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Vesting</span>
                  <span className="text-white font-semibold">{t.vesting}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
