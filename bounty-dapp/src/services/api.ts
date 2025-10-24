import axios from 'axios'

const API_BASE_URL = import.meta.env?.VITE_API_URL || 'https://bounty.quillon.xyz'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface RegisterRequest {
  testnet_address: string
  mainnet_address?: string
}

export interface RegisterResponse {
  user_id: string
  message: string
}

export interface UserScore {
  user_id: string
  total_score: number
  rank: number | null
  tier: string
  category_scores: CategoryScores
  early_multiplier: number
  consistency_bonus: number
}

export interface CategoryScores {
  node_ops: number
  transactions: number
  bug_reports: number
  community: number
  social: number
}

export interface LeaderboardEntry {
  rank: number
  testnet_address: string
  total_score: number
  tier: string
  category_scores: CategoryScores
}

export interface BugReportRequest {
  user_id: string
  github_issue_url: string
  severity: 'Critical' | 'High' | 'Medium' | 'Low'
  description: string
}

export interface BugReportResponse {
  report_id: string
  points_awarded: number
  message: string
}

export interface SocialActivityRequest {
  user_id: string
  platform: 'twitter' | 'github' | 'discord' | 'medium' | 'youtube'
  activity_url: string
  activity_type: 'Tweet' | 'Thread' | 'Article' | 'Video' | 'DiscordMessage' | 'GitHubPR' | 'GitHubIssue'
}

export interface SocialActivityResponse {
  message: string
  base_points: number
  status: string
}

// API functions
export const bountyApi = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health')
    return response.data
  },

  // Register a new user
  async register(data: RegisterRequest): Promise<RegisterResponse> {
    const response = await api.post('/v1/testnet/register', data)
    return response.data
  },

  // Get user score
  async getUserScore(userId: string): Promise<UserScore> {
    const response = await api.get(`/v1/testnet/score/${userId}`)
    return response.data
  },

  // Get leaderboard
  async getLeaderboard(limit: number = 100): Promise<LeaderboardEntry[]> {
    const response = await api.get('/v1/testnet/leaderboard', {
      params: { limit },
    })
    return response.data
  },

  // Submit bug report
  async submitBugReport(data: BugReportRequest): Promise<BugReportResponse> {
    const response = await api.post('/v1/testnet/bug-report', data)
    return response.data
  },

  // Submit social activity
  async submitSocialActivity(data: SocialActivityRequest): Promise<SocialActivityResponse> {
    const response = await api.post('/v1/testnet/social-activity', data)
    return response.data
  },
}

export default bountyApi
