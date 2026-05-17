import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Bug, AlertCircle, Check, AlertTriangle } from 'lucide-react'
import bountyApi from '../services/api'

export default function BugReports() {
  const [userId, setUserId] = useState('')
  const [issueUrl, setIssueUrl] = useState('')
  const [severity, setSeverity] = useState<'Critical' | 'High' | 'Medium' | 'Low'>('Medium')
  const [description, setDescription] = useState('')
  const [success, setSuccess] = useState(false)
  const [reportId, setReportId] = useState('')
  const [pointsAwarded, setPointsAwarded] = useState(0)

  const submitMutation = useMutation({
    mutationFn: (data: { user_id: string; issue_url: string; severity: string; description: string }) => bountyApi.submitBugReport(data as any),
    onSuccess: (data) => {
      setSuccess(true)
      setReportId(data.report_id)
      setPointsAwarded(data.points_awarded)
      setIssueUrl('')
      setDescription('')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitMutation.mutate({
      user_id: userId,
      issue_url: issueUrl,
      severity,
      description,
    })
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 mb-4">
          Submit Bug Reports
        </h1>
        <p className="text-slate-300 text-lg">
          Help improve Q-NarwhalKnight and earn rewards for verified bug reports
        </p>
      </div>

      {success && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-6 flex items-start space-x-3">
          <Check className="w-6 h-6 text-green-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-green-400 font-semibold mb-2">Bug Report Submitted!</h3>
            <div className="space-y-2">
              <p className="text-slate-300">Report ID: <code className="text-green-400">{reportId}</code></p>
              <p className="text-slate-300">Base Points Awarded: <span className="font-bold">{pointsAwarded}</span></p>
              <p className="text-slate-400 text-sm">
                Your report is pending verification. Points will be finalized after review.
              </p>
            </div>
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
                Issue / PR URL <span className="text-red-400">*</span>
              </label>
              <input
                type="url"
                value={issueUrl}
                onChange={(e) => setIssueUrl(e.target.value)}
                placeholder="https://code.quillon.xyz/issues/123 or branch URL"
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                required
              />
              <p className="text-xs text-slate-400 mt-1">
                Link to your issue or merge request on code.quillon.xyz
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Severity <span className="text-red-400">*</span>
              </label>
              <select
                value={severity}
                onChange={(e) => setSeverity(e.target.value as any)}
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="Low">Low</option>
                <option value="Medium">Medium</option>
                <option value="High">High</option>
                <option value="Critical">Critical</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Description <span className="text-red-400">*</span>
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the bug, steps to reproduce, and impact..."
                rows={6}
                className="w-full px-4 py-3 bg-slate-900/50 border border-purple-500/30 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                required
              />
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
                  <Bug className="w-5 h-5" />
                  <span>Submit Bug Report</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Severity Guide */}
        <div className="space-y-6">
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-yellow-400" />
              Severity Levels
            </h3>
            <div className="space-y-4">
              <SeverityLevel
                level="Critical"
                points={100}
                color="text-red-400"
                description="System crash, data loss, security breach"
              />
              <SeverityLevel
                level="High"
                points={50}
                color="text-orange-400"
                description="Major feature broken, significant impact"
              />
              <SeverityLevel
                level="Medium"
                points={20}
                color="text-yellow-400"
                description="Feature not working as expected"
              />
              <SeverityLevel
                level="Low"
                points={10}
                color="text-blue-400"
                description="Minor issue, cosmetic bug"
              />
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-2">Tips for Good Reports</h4>
            <ul className="text-sm text-slate-300 space-y-1 ml-4 list-disc">
              <li>Provide clear steps to reproduce</li>
              <li>Include error messages/logs</li>
              <li>Describe expected vs actual behavior</li>
              <li>Mention your environment (OS, version)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

interface SeverityLevelProps {
  level: string
  points: number
  color: string
  description: string
}

function SeverityLevel({ level, points, color, description }: SeverityLevelProps) {
  return (
    <div className="pb-4 border-b border-slate-700 last:border-b-0 last:pb-0">
      <div className="flex items-center justify-between mb-1">
        <span className={`font-semibold ${color}`}>{level}</span>
        <span className="text-white font-bold">{points} pts</span>
      </div>
      <p className="text-xs text-slate-400">{description}</p>
    </div>
  )
}
