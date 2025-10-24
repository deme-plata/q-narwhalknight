import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { UserPlus, Check, AlertCircle } from 'lucide-react'
import bountyApi, { type RegisterRequest } from '../services/api'

export default function Register() {
  const [testnetAddress, setTestnetAddress] = useState('')
  const [mainnetAddress, setMainnetAddress] = useState('')
  const [success, setSuccess] = useState(false)
  const [userId, setUserId] = useState('')

  const registerMutation = useMutation({
    mutationFn: (data: RegisterRequest) => bountyApi.register(data),
    onSuccess: (data) => {
      setSuccess(true)
      setUserId(data.user_id)
      setTestnetAddress('')
      setMainnetAddress('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!testnetAddress.trim()) return

    registerMutation.mutate({
      testnet_address: testnetAddress.trim(),
      mainnet_address: mainnetAddress.trim() || undefined,
    })
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 mb-4">
          Register for Testnet Bounty
        </h1>
        <p className="text-slate-300 text-lg">
          Join the Q-NarwhalKnight testnet campaign and start earning rewards
        </p>
      </div>

      {success && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-6 mb-6 flex items-start space-x-3">
          <Check className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-green-400 font-semibold mb-2">
              {registerMutation.data?.message || 'Registration Successful!'}
            </h3>
            <p className="text-slate-300 mb-2">Your User ID:</p>
            <code className="block bg-slate-900/50 px-4 py-2 rounded-lg text-green-400 font-mono text-sm break-all">
              {userId}
            </code>
            <p className="text-slate-400 text-sm mt-3">
              💡 Save this ID! You'll need it to track your bounty progress and claim rewards.
            </p>
            <p className="text-slate-400 text-sm mt-2">
              📋 Tip: Copy this ID now and store it safely. You can bookmark this page or take a screenshot.
            </p>
          </div>
        </div>
      )}

      {registerMutation.isError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-6 flex items-start space-x-3">
          <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-red-400 font-semibold mb-2">Registration Failed</h3>
            <p className="text-slate-300">
              {(registerMutation.error as Error)?.message || 'An error occurred during registration'}
            </p>
          </div>
        </div>
      )}

      <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-8 border border-purple-500/30">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Testnet Address */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Testnet Wallet Address <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={testnetAddress}
              onChange={(e) => setTestnetAddress(e.target.value)}
              placeholder="Enter your testnet wallet address (qnk... or 64 hex chars)"
              className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              required
              pattern="(qnk[0-9a-fA-F]{64}|[0-9a-fA-F]{64})"
              title="Must be a 64-character hexadecimal address (optionally prefixed with 'qnk')"
            />
            <p className="text-xs text-slate-400 mt-1">
              Address from your Q-NarwhalKnight testnet wallet (with or without 'qnk' prefix)
            </p>
          </div>

          {/* Mainnet Address */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Mainnet Wallet Address (Optional)
            </label>
            <input
              type="text"
              value={mainnetAddress}
              onChange={(e) => setMainnetAddress(e.target.value)}
              placeholder="Enter your mainnet wallet address for reward claims"
              className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              pattern="(qnk[0-9a-fA-F]{64}|[0-9a-fA-F]{64})"
              title="Must be a 64-character hexadecimal address (optionally prefixed with 'qnk')"
            />
            <p className="text-xs text-slate-400 mt-1">
              Link your mainnet wallet now for seamless reward distribution (can be added later)
            </p>
          </div>

          {/* Info Section */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-2 flex items-center">
              <UserPlus className="w-5 h-5 mr-2" />
              What happens next?
            </h4>
            <ul className="text-sm text-slate-300 space-y-2 ml-7">
              <li>• You'll receive a unique User ID to track your bounty progress</li>
              <li>• Start participating in testnet activities to earn points</li>
              <li>• Your score is calculated across 5 categories with weighted distribution</li>
              <li>• Early participants receive bonus multipliers up to 2x</li>
              <li>• Final rewards are distributed to mainnet based on your tier and score</li>
            </ul>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={registerMutation.isPending}
            className="w-full px-6 py-4 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
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

      {/* Tier Information */}
      <div className="mt-8 bg-slate-800/50 backdrop-blur-lg rounded-xl p-8 border border-purple-500/30">
        <h3 className="text-2xl font-semibold text-white mb-6">Bounty Tier System</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <TierCard
            tier="Pioneer"
            percentage="20%"
            rank="Top 1%"
            vesting="6 months"
            color="from-yellow-400 to-yellow-600"
          />
          <TierCard
            tier="Contributor"
            percentage="30%"
            rank="Top 10%"
            vesting="3 months"
            color="from-blue-400 to-blue-600"
          />
          <TierCard
            tier="Participant"
            percentage="40%"
            rank="Top 50%"
            vesting="1 month"
            color="from-cyan-400 to-cyan-600"
          />
          <TierCard
            tier="Supporter"
            percentage="10%"
            rank="All valid"
            vesting="None"
            color="from-purple-400 to-purple-600"
          />
        </div>
      </div>
    </div>
  )
}

interface TierCardProps {
  tier: string
  percentage: string
  rank: string
  vesting: string
  color: string
}

function TierCard({ tier, percentage, rank, vesting, color }: TierCardProps) {
  return (
    <div className="bg-slate-900/50 rounded-lg p-4 border border-purple-500/20">
      <div className={`inline-flex px-3 py-1 rounded-full bg-gradient-to-r ${color} text-white text-sm font-semibold mb-3`}>
        {tier}
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-400">Pool:</span>
          <span className="text-white font-semibold">{percentage}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Rank:</span>
          <span className="text-white font-semibold">{rank}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Vesting:</span>
          <span className="text-white font-semibold">{vesting}</span>
        </div>
      </div>
    </div>
  )
}
