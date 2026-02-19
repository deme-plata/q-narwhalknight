import { useQuery } from '@tanstack/react-query'
import { Trophy, Medal, Award } from 'lucide-react'
import bountyApi from '../services/api'

export default function Leaderboard() {
  const { data: leaderboard, isLoading } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: () => bountyApi.getLeaderboard(100),
  })

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 mb-4">
          Bounty Leaderboard
        </h1>
        <p className="text-slate-300 text-lg">
          Top contributors in the Q-NarwhalKnight mainnet bounty campaign
        </p>
      </div>

      {isLoading && (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      )}

      {leaderboard && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl border border-purple-500/30 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-900/50">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-slate-300">Rank</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-slate-300">Address</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-slate-300">Tier</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Total Score</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Node Ops</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Transactions</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Bug Reports</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Community</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-slate-300">Social</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/30">
                {leaderboard.map((entry, index) => (
                  <tr
                    key={index}
                    className="hover:bg-slate-700/20 transition-colors duration-200"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        {entry.rank === 1 && <Trophy className="w-5 h-5 text-yellow-400" />}
                        {entry.rank === 2 && <Medal className="w-5 h-5 text-gray-400" />}
                        {entry.rank === 3 && <Award className="w-5 h-5 text-orange-400" />}
                        <span className={`font-bold ${
                          entry.rank <= 3 ? 'text-white' : 'text-slate-300'
                        }`}>
                          #{entry.rank}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <code className="text-sm text-slate-300 font-mono">
                        {typeof entry.testnet_address === 'string'
                          ? `${entry.testnet_address.slice(0, 10)}...${entry.testnet_address.slice(-6)}`
                          : `qnk${(entry.testnet_address as number[]).map(b => b.toString(16).padStart(2,'0')).join('').slice(0,6)}...`
                        }
                      </code>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold ${
                        entry.tier === 'Pioneer' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                        entry.tier === 'Contributor' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                        entry.tier === 'Participant' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' :
                        'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                      }`}>
                        {entry.tier}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span className="font-bold text-white">{entry.total_score.toFixed(2)}</span>
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">
                      {entry.category_scores.node_ops.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">
                      {entry.category_scores.transactions.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">
                      {entry.category_scores.bug_reports.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">
                      {entry.category_scores.community.toFixed(1)}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-300">
                      {entry.category_scores.social.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
