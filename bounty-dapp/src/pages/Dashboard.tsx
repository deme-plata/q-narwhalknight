import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js'
import { Doughnut, Bar } from 'react-chartjs-2'
import { Trophy, TrendingUp, Award, Users } from 'lucide-react'
import bountyApi from '../services/api'

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title)

export default function Dashboard() {
  const [userId, setUserId] = useState('')
  const [searchUserId, setSearchUserId] = useState('')

  const { data: userScore, isLoading, error } = useQuery({
    queryKey: ['userScore', searchUserId],
    queryFn: () => bountyApi.getUserScore(searchUserId),
    enabled: !!searchUserId,
  })

  const { data: leaderboard } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: () => bountyApi.getLeaderboard(10),
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (userId.trim()) {
      setSearchUserId(userId.trim())
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 mb-4">
          Testnet Bounty Dashboard
        </h1>
        <p className="text-slate-300 text-lg">
          Track your contributions and earn rewards for helping build Q-NarwhalKnight
        </p>
      </div>

      {/* Search Section */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
        <form onSubmit={handleSearch} className="flex gap-4">
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter your User ID to view stats..."
            className="flex-1 px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <button
            type="submit"
            className="px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200"
          >
            Search
          </button>
        </form>
      </div>

      {/* User Score Section */}
      {isLoading && (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-red-400">
          Failed to load user data. Please check the User ID and try again.
        </div>
      )}

      {userScore && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <StatCard
              icon={<Trophy className="w-8 h-8" />}
              title="Total Score"
              value={userScore.total_score.toFixed(2)}
              color="from-purple-500 to-blue-500"
            />
            <StatCard
              icon={<Award className="w-8 h-8" />}
              title="Rank"
              value={userScore.rank ? `#${userScore.rank}` : 'Unranked'}
              color="from-blue-500 to-cyan-500"
            />
            <StatCard
              icon={<TrendingUp className="w-8 h-8" />}
              title="Tier"
              value={userScore.tier}
              color="from-cyan-500 to-teal-500"
            />
            <StatCard
              icon={<Users className="w-8 h-8" />}
              title="Early Bonus"
              value={`${userScore.early_multiplier.toFixed(2)}x`}
              color="from-teal-500 to-green-500"
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Category Breakdown - Doughnut */}
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
              <h3 className="text-xl font-semibold text-white mb-4">Score Breakdown</h3>
              <div className="h-64 flex items-center justify-center">
                <Doughnut
                  data={{
                    labels: ['Node Ops', 'Transactions', 'Bug Reports', 'Community', 'Social'],
                    datasets: [
                      {
                        data: [
                          userScore.category_scores.node_ops,
                          userScore.category_scores.transactions,
                          userScore.category_scores.bug_reports,
                          userScore.category_scores.community,
                          userScore.category_scores.social,
                        ],
                        backgroundColor: [
                          'rgba(147, 51, 234, 0.8)',
                          'rgba(59, 130, 246, 0.8)',
                          'rgba(6, 182, 212, 0.8)',
                          'rgba(16, 185, 129, 0.8)',
                          'rgba(236, 72, 153, 0.8)',
                        ],
                        borderColor: [
                          'rgba(147, 51, 234, 1)',
                          'rgba(59, 130, 246, 1)',
                          'rgba(6, 182, 212, 1)',
                          'rgba(16, 185, 129, 1)',
                          'rgba(236, 72, 153, 1)',
                        ],
                        borderWidth: 2,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: {
                          color: '#f1f5f9',
                          padding: 15,
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>

            {/* Category Scores - Bar Chart */}
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
              <h3 className="text-xl font-semibold text-white mb-4">Category Scores</h3>
              <div className="h-64">
                <Bar
                  data={{
                    labels: ['Node Ops', 'Transactions', 'Bug Reports', 'Community', 'Social'],
                    datasets: [
                      {
                        label: 'Score',
                        data: [
                          userScore.category_scores.node_ops,
                          userScore.category_scores.transactions,
                          userScore.category_scores.bug_reports,
                          userScore.category_scores.community,
                          userScore.category_scores.social,
                        ],
                        backgroundColor: [
                          'rgba(147, 51, 234, 0.8)',
                          'rgba(59, 130, 246, 0.8)',
                          'rgba(6, 182, 212, 0.8)',
                          'rgba(16, 185, 129, 0.8)',
                          'rgba(236, 72, 153, 0.8)',
                        ],
                        borderColor: [
                          'rgba(147, 51, 234, 1)',
                          'rgba(59, 130, 246, 1)',
                          'rgba(6, 182, 212, 1)',
                          'rgba(16, 185, 129, 1)',
                          'rgba(236, 72, 153, 1)',
                        ],
                        borderWidth: 2,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        ticks: { color: '#f1f5f9' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                      },
                      x: {
                        ticks: { color: '#f1f5f9' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' },
                      },
                    },
                    plugins: {
                      legend: {
                        display: false,
                      },
                    },
                  }}
                />
              </div>
            </div>
          </div>

          {/* Multipliers */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            <h3 className="text-xl font-semibold text-white mb-4">Bonus Multipliers</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 p-4 rounded-lg border border-purple-500/20">
                <div className="text-slate-400 mb-1">Early Participation Multiplier</div>
                <div className="text-2xl font-bold text-white">{userScore.early_multiplier.toFixed(2)}x</div>
                <div className="text-sm text-slate-500 mt-2">Rewards early adopters with up to 2x bonus</div>
              </div>
              <div className="bg-gradient-to-r from-cyan-500/10 to-teal-500/10 p-4 rounded-lg border border-cyan-500/20">
                <div className="text-slate-400 mb-1">Consistency Bonus</div>
                <div className="text-2xl font-bold text-white">{userScore.consistency_bonus.toFixed(2)}x</div>
                <div className="text-sm text-slate-500 mt-2">Daily activity bonus up to 1.2x</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top Contributors */}
      {leaderboard && leaderboard.length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
          <h3 className="text-2xl font-semibold text-white mb-6">Top 10 Contributors</h3>
          <div className="space-y-3">
            {leaderboard.map((entry, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-purple-500/20 hover:border-purple-500/40 transition-all duration-200"
              >
                <div className="flex items-center space-x-4">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                      index === 0
                        ? 'bg-gradient-to-r from-yellow-400 to-yellow-600 text-yellow-900'
                        : index === 1
                        ? 'bg-gradient-to-r from-gray-300 to-gray-500 text-gray-900'
                        : index === 2
                        ? 'bg-gradient-to-r from-orange-400 to-orange-600 text-orange-900'
                        : 'bg-slate-700 text-slate-300'
                    }`}
                  >
                    #{entry.rank}
                  </div>
                  <div>
                    <div className="font-mono text-sm text-slate-300">
                      {entry.testnet_address.slice(0, 8)}...{entry.testnet_address.slice(-6)}
                    </div>
                    <div className="text-xs text-slate-500">{entry.tier}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-white">{entry.total_score.toFixed(2)}</div>
                  <div className="text-xs text-slate-400">points</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface StatCardProps {
  icon: React.ReactNode
  title: string
  value: string
  color: string
}

function StatCard({ icon, title, value, color }: StatCardProps) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30 hover:border-purple-500/50 transition-all duration-200">
      <div className={`inline-flex p-3 rounded-lg bg-gradient-to-r ${color} mb-4`}>
        {icon}
      </div>
      <div className="text-slate-400 text-sm mb-1">{title}</div>
      <div className="text-3xl font-bold text-white">{value}</div>
    </div>
  )
}
