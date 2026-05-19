// Shared types for the Twitter MCP — match the wire contract of the
// Rust x-algorithm-scorer sidecar at tools/quillon-twitter-mcp/crates/.

export interface ScoreContext {
  recent_tweets?: string[];
  target_audience?: string;
}

export interface PerActionProbabilities {
  favorite: number;
  reply: number;
  repost: number;
  quote: number;
  click: number;
  profile_visit: number;
  video_view: number;
  photo_expand: number;
  share: number;
  dwell_time: number;
  block: number;
  mute: number;
  report: number;
}

export interface VariantSuggestion {
  text: string;
  score_delta: number;
  why: string;
}

export interface ScoreResponse {
  predicted_engagement: number;
  negative_signal_risk: number;
  per_action_probabilities: PerActionProbabilities;
  variant_suggestions: VariantSuggestion[];
  model_version: string;
}

// Status lifecycle for a queued draft
export type DraftStatus = 'pending' | 'approved' | 'rejected' | 'published' | 'expired';

export interface QueuedDraft {
  id: string; // UUID
  draft_text: string;
  author_wallet?: string; // qnk... address that drafted this
  score: ScoreResponse | null;
  status: DraftStatus;
  created_at: string; // ISO 8601
  approved_at: string | null;
  approved_by: string | null; // admin wallet
  approval_signature: string | null; // hex Ed25519 signature
  references_tx?: string; // optional referenced on-chain tx hash
  references_pr?: number; // optional referenced GitHub PR number
}

// AFL-1 §5.3 hard limit on negative-signal risk for queued drafts.
// Configurable via Q_TWITTER_NEG_RISK_MAX env var; default 0.15.
export const DEFAULT_NEG_RISK_MAX = 0.15;
