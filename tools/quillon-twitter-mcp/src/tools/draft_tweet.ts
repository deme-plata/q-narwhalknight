// draft_tweet MCP tool — returns a draft text + immediately calls
// score_draft to attach predicted engagement.
//
// IMPORTANT: this tool does NOT call the Anthropic API to generate text.
// The MCP server is consumed by an AI agent (Claude Code, Codex, Grok
// Build) — the agent already has generation capability. This tool just
// formalises "I have this draft" + scores it in one round-trip, and
// returns suggestions if neg-signal-risk is too high.
//
// The agent decides whether to call `queue_for_approval` next.

import { z } from 'zod';
import { scoreDraft, ScorerUnavailableError } from '../scorer_client.js';
import type { ScoreContext, ScoreResponse } from '../types.js';
import { DEFAULT_NEG_RISK_MAX } from '../types.js';

export const draftTweetSchema = {
  draft_text: z
    .string()
    .min(1)
    .max(280)
    .describe('Your draft tweet text. X allows up to 280 chars; respect that.'),
  references_tx: z
    .string()
    .optional()
    .describe('Optional: on-chain transaction this tweet references (for later attestation)'),
  references_pr: z
    .number()
    .int()
    .optional()
    .describe('Optional: GitHub PR number this tweet references'),
  topic: z
    .string()
    .optional()
    .describe('Topic hint to help the scorer (e.g. "agentic AI", "release announcement")'),
};

const NEG_RISK_MAX = parseFloat(
  process.env.Q_TWITTER_NEG_RISK_MAX || `${DEFAULT_NEG_RISK_MAX}`,
);

export async function draftTweetHandler(args: {
  draft_text: string;
  references_tx?: string;
  references_pr?: number;
  topic?: string;
}): Promise<{ content: Array<{ type: 'text'; text: string }> }> {
  const context: ScoreContext = {};
  if (args.topic) context.target_audience = args.topic;

  let score: ScoreResponse | null = null;
  let scorerError: string | null = null;
  try {
    score = await scoreDraft(args.draft_text, context);
  } catch (e) {
    scorerError =
      e instanceof ScorerUnavailableError ? e.message : (e as Error).message ?? 'unknown';
  }

  const lines = [`=== Draft ===`, '', args.draft_text, '', `Length: ${args.draft_text.length} chars`];

  if (args.references_tx) lines.push(`References tx: ${args.references_tx}`);
  if (args.references_pr) lines.push(`References PR: #${args.references_pr}`);

  if (!score) {
    lines.push(
      '',
      `⚠️  Scorer unavailable: ${scorerError}`,
      `Continuing without scoring. Use \`queue_for_approval\` to persist this draft anyway,`,
      `but note that AFL-1 §5.3 negative-signal-risk hard limit cannot be enforced.`,
    );
  } else {
    lines.push(
      '',
      `Predicted engagement: ${(score.predicted_engagement * 100).toFixed(0)}%`,
      `Negative signal risk: ${(score.negative_signal_risk * 100).toFixed(0)}%${
        score.negative_signal_risk > NEG_RISK_MAX ? ` ⚠️  ABOVE ${NEG_RISK_MAX * 100}% LIMIT` : ' ✓'
      }`,
    );

    if (score.variant_suggestions.length > 0) {
      lines.push('', `Suggestions to improve score:`);
      for (const [i, v] of score.variant_suggestions.entries()) {
        lines.push(`  ${i + 1}. ${v.text}`);
        lines.push(`     (+${(v.score_delta * 100).toFixed(0)}% — ${v.why})`);
      }
    }

    if (score.negative_signal_risk > NEG_RISK_MAX) {
      lines.push(
        '',
        `🚫 This draft EXCEEDS the negative-signal-risk hard limit (${NEG_RISK_MAX * 100}%).`,
        `   queue_for_approval will REFUSE to queue it as-is.`,
        `   Revise the draft (use one of the suggestions above) and try again.`,
      );
    } else {
      lines.push('', `✓ Within tolerances. Call \`queue_for_approval\` to submit for admin review.`);
    }
  }

  return { content: [{ type: 'text', text: lines.join('\n') }] };
}
