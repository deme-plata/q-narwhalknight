import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Share2, Twitter, Code2, MessageCircle, Youtube, FileText, Check, AlertCircle } from 'lucide-react'
import bountyApi from '../services/api'

export default function SocialActivity() {
  const [userId, setUserId] = useState('')
  const [platform, setPlatform] = useState<'twitter' | 'code_quillon' | 'discord' | 'medium' | 'youtube'>('twitter')
  const [activityUrl, setActivityUrl] = useState('')
  const [activityType, setActivityType] = useState<string>('Tweet')
  const [success, setSuccess] = useState(false)
  const [basePoints, setBasePoints] = useState(0)

  const submitMutation = useMutation({
    mutationFn: (data: any) => bountyApi.submitSocialActivity(data),
    onSuccess: (data) => {
      setSuccess(true)
      setBasePoints(data.base_points)
      setActivityUrl('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitMutation.mutate({
      user_id: userId,
      platform,
      activity_url: activityUrl,
      activity_type: activityType,
    })
  }

  const platformIcon = {
    twitter: <Twitter className="w-5 h-5" />,
    code_quillon: <Code2 className="w-5 h-5" />,
    discord: <MessageCircle className="w-5 h-5" />,
    medium: <FileText className="w-5 h-5" />,
    youtube: <Youtube className="w-5 h-5" />,
  }

  const platformLabels: Record<string, string> = {
    twitter: 'Twitter',
    code_quillon: 'Code',
    discord: 'Discord',
    medium: 'Medium',
    youtube: 'YouTube',
  }

  const activityTypesByPlatform = {
    twitter: ['Tweet', 'Thread'],
    code_quillon: ['MergeRequest', 'CodeIssue'],
    discord: ['DiscordMessage'],
    medium: ['Article'],
    youtube: ['Video'],
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 mb-4">
          Social Media Activity
        </h1>
        <p className="text-slate-300 text-lg">
          Share Q-NarwhalKnight content and earn rewards for community engagement
        </p>
      </div>

      {success && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-6 flex items-start space-x-3">
          <Check className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-green-400 font-semibold mb-2">Activity Submitted!</h3>
            <p className="text-slate-300">Base Points: <span className="font-bold">{basePoints}</span></p>
            <p className="text-slate-400 text-sm mt-2">
              Your submission is pending verification. Final points will be calculated based on engagement metrics.
            </p>
          </div>
        </div>
      )}

      {submitMutation.isError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 flex items-start space-x-3">
          <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-red-400 font-semibold mb-2">Submission Failed</h3>
            <p className="text-slate-300">
              {(submitMutation.error as Error)?.message || 'An error occurred'}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Submit Form */}
        <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-lg rounded-xl p-8 border border-purple-500/30">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Your User ID <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter your bounty user ID"
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Platform <span className="text-red-400">*</span>
              </label>
              <div className="grid grid-cols-5 gap-2">
                {(['twitter', 'code_quillon', 'discord', 'medium', 'youtube'] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => {
                      setPlatform(p)
                      setActivityType(activityTypesByPlatform[p][0] as any)
                    }}
                    className={`flex flex-col items-center justify-center p-3 rounded-lg border-2 transition-all duration-200 ${
                      platform === p
                        ? 'border-purple-500 bg-purple-500/20 text-white'
                        : 'border-slate-700 bg-slate-900/50 text-slate-400 hover:border-purple-500/50'
                    }`}
                  >
                    {platformIcon[p]}
                    <span className="text-xs mt-1">{platformLabels[p] || p}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Activity Type <span className="text-red-400">*</span>
              </label>
              <select
                value={activityType}
                onChange={(e) => setActivityType(e.target.value as any)}
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {activityTypesByPlatform[platform].map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Content URL <span className="text-red-400">*</span>
              </label>
              <input
                type="url"
                value={activityUrl}
                onChange={(e) => setActivityUrl(e.target.value)}
                placeholder="https://..."
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                required
              />
              <p className="text-xs text-slate-400 mt-1">
                Direct link to your {platform} post, article, or video
              </p>
            </div>

            <button
              type="submit"
              disabled={submitMutation.isPending}
              className="w-full px-6 py-4 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-blue-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {submitMutation.isPending ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                  <span>Submitting...</span>
                </>
              ) : (
                <>
                  <Share2 className="w-5 h-5" />
                  <span>Submit Activity</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Activity Point Values */}
        <div className="space-y-6">
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            <h3 className="text-xl font-semibold text-white mb-4">Base Points</h3>
            <div className="space-y-3">
              <ActivityPoints type="Video" points={50} />
              <ActivityPoints type="Article" points={40} />
              <ActivityPoints type="Merge Request" points={35} />
              <ActivityPoints type="Thread" points={30} />
              <ActivityPoints type="Code Issue" points={15} />
              <ActivityPoints type="Tweet" points={10} />
              <ActivityPoints type="Discord Message" points={5} />
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-2">Engagement Multiplier</h4>
            <p className="text-sm text-slate-300 mb-2">
              Final points are calculated based on:
            </p>
            <ul className="text-sm text-slate-300 space-y-1 ml-4 list-disc">
              <li>Likes/Hearts</li>
              <li>Shares/Retweets</li>
              <li>Comments/Replies</li>
              <li>Views (for videos)</li>
            </ul>
          </div>

          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2">Content Guidelines</h4>
            <ul className="text-sm text-slate-300 space-y-1 ml-4 list-disc">
              <li>Must mention Q-NarwhalKnight</li>
              <li>Genuine, informative content</li>
              <li>No spam or duplicate posts</li>
              <li>Follow platform rules</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

interface ActivityPointsProps {
  type: string
  points: number
}

function ActivityPoints({ type, points }: ActivityPointsProps) {
  return (
    <div className="flex items-center justify-between pb-3 border-b border-slate-700 last:border-b-0 last:pb-0">
      <span className="text-slate-300 text-sm">{type}</span>
      <span className="text-white font-bold">{points} pts</span>
    </div>
  )
}
