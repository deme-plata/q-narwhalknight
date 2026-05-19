// score_draft MCP tool — calls the Rust sidecar and returns the score
// in a human-readable text format suitable for an AI agent or user
// reading the result in their MCP client.

import { z } from 'zod';
import { scoreDraft, ScorerUnavailableError, scorerVersion } from '../scorer_client.js';
import type { ScoreContext } from '../types.js';

export const scoreDraftSchema = {
  draft_text: z.string().min(1).max(32000).describe('Tweet text to score'),
  recent_tweets: z
    .array(z.string())
    .optional()
    .describe('Author style calibration: 3-5 recent tweets from the same author'),
  target_audience: z
    .string()
    .optional()
    .describe('e.g. "developers", "crypto-traders", "general"'),
};

export async function scoreDraftHandler(args: {
  draft_text: string;
  recent_tweets?: string[];
  target_audience?: string;
}): Promise<{ content: Array<{ type: 'text'; text: string }> }> {
  const context: ScoreContext = {};
  if (args.recent_tweets) context.recent_tweets = args.recent_tweets;
  if (args.target_audience) context.target_audience = args.target_audience;

  let scoreResult;
  try {
    scoreResult = await scoreDraft(args.draft_text, context);
  } catch (e) {
    if (e instanceof ScorerUnavailableError) {
      return {
        content: [
          {
            type: 'text',
            text: `Scorer sidecar unreachable. Start it with:\n\n  cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer && cargo run --release\n\nError: ${e.message}`,
          },
        ],
      };
    }
    throw e;
  }

  const v = await scorerVersion();
  const versionStr = v ? `${v.model} (Layer ${v.layer})` : 'unknown';

  const lines = [
    `=== Tweet score (${versionStr}) ===`,
    '',
    `Predicted engagement:  ${(scoreResult.predicted_engagement * 100).toFixed(1)}%`,
    `Negative signal risk:  ${(scoreResult.negative_signal_risk * 100).toFixed(1)}%   ${
      scoreResult.negative_signal_risk > 0.15
        ? '⚠️  ABOVE 0.15 HARD LIMIT — refuse to queue per AFL-1 §5.3'
        : '✓ within tolerance'
    }`,
    '',
    'Per-action probabilities:',
    `  favorite:      ${pct(scoreResult.per_action_probabilities.favorite)}`,
    `  reply:         ${pct(scoreResult.per_action_probabilities.reply)}`,
    `  repost:        ${pct(scoreResult.per_action_probabilities.repost)}`,
    `  quote:         ${pct(scoreResult.per_action_probabilities.quote)}`,
    `  click:         ${pct(scoreResult.per_action_probabilities.click)}`,
    `  profile visit: ${pct(scoreResult.per_action_probabilities.profile_visit)}`,
    `  share:         ${pct(scoreResult.per_action_probabilities.share)}`,
    `  dwell time:    ${pct(scoreResult.per_action_probabilities.dwell_time)}`,
    `  block:         ${pct(scoreResult.per_action_probabilities.block)}   (negative)`,
    `  mute:          ${pct(scoreResult.per_action_probabilities.mute)}   (negative)`,
    `  report:        ${pct(scoreResult.per_action_probabilities.report)}   (negative)`,
  ];

  if (scoreResult.variant_suggestions.length > 0) {
    lines.push('', `Variant suggestions (${scoreResult.variant_suggestions.length}):`);
    for (const [i, v] of scoreResult.variant_suggestions.entries()) {
      lines.push(`  ${i + 1}. ${v.text}`);
      lines.push(`     score Δ: +${(v.score_delta * 100).toFixed(1)}%  — ${v.why}`);
    }
  }

  lines.push('', 'Next: call `queue_for_approval` if you want this drafted for admin approval.');

  return { content: [{ type: 'text', text: lines.join('\n') }] };
}

function pct(x: number): string {
  return `${(x * 100).toFixed(0).padStart(3)}%`;
}
