// queue_for_approval MCP tool — persists a draft into SQLite + returns
// an approval URL the admin wallet must visit to sign-approve.
//
// AFL-1 §5.3 hard limit: refuse to queue if negative-signal-risk > 0.15.
// Configurable via Q_TWITTER_NEG_RISK_MAX env var.

import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { insertDraft } from '../db.js';
import { scoreDraft, ScorerUnavailableError } from '../scorer_client.js';
import { DEFAULT_NEG_RISK_MAX } from '../types.js';

const APPROVAL_BASE_URL =
  process.env.QTWITTER_APPROVAL_BASE_URL || 'https://quillon.xyz/admin/twitter/q';

const NEG_RISK_MAX = parseFloat(
  process.env.Q_TWITTER_NEG_RISK_MAX || `${DEFAULT_NEG_RISK_MAX}`,
);

export const queueForApprovalSchema = {
  draft_text: z
    .string()
    .min(1)
    .max(280)
    .describe('The tweet draft text. Max 280 chars per X API.'),
  author_wallet: z
    .string()
    .regex(/^qnk[0-9a-f]{64}$/)
    .optional()
    .describe('Your qnk... address (for filtering /list_my_drafts later). Optional.'),
  references_tx: z
    .string()
    .regex(/^[0-9a-f]{64}$/)
    .optional()
    .describe('Optional on-chain transaction hash this tweet references — enables chain attestation on publish'),
  references_pr: z
    .number()
    .int()
    .positive()
    .optional()
    .describe('Optional GitHub PR number this tweet references'),
  skip_score: z
    .boolean()
    .optional()
    .describe('If true, skip the scorer call. Use ONLY if the scorer sidecar is known to be down AND you accept that the neg-signal-risk hard limit cannot be enforced.'),
};

export async function queueForApprovalHandler(args: {
  draft_text: string;
  author_wallet?: string;
  references_tx?: string;
  references_pr?: number;
  skip_score?: boolean;
}): Promise<{ content: Array<{ type: 'text'; text: string }> }> {
  let score = null;
  let scoreErrorMsg: string | null = null;

  if (!args.skip_score) {
    try {
      score = await scoreDraft(args.draft_text);
    } catch (e) {
      scoreErrorMsg =
        e instanceof ScorerUnavailableError
          ? `Scorer sidecar unreachable: ${e.message}`
          : (e as Error).message ?? 'unknown';
    }

    if (score && score.negative_signal_risk > NEG_RISK_MAX) {
      return {
        content: [
          {
            type: 'text',
            text: [
              `🚫 REFUSED TO QUEUE — negative-signal-risk hard limit exceeded`,
              ``,
              `  Predicted neg-signal-risk: ${(score.negative_signal_risk * 100).toFixed(1)}%`,
              `  Hard limit (per AFL-1 §5.3): ${NEG_RISK_MAX * 100}%`,
              ``,
              `Per the Twitter MCP spec, the system refuses to queue drafts that`,
              `the algorithm predicts would attract block/mute/report responses.`,
              `This is the anti-embarrassment hard limit — even the admin cannot`,
              `override it without revising the draft.`,
              ``,
              `Variant suggestions (call \`draft_tweet\` for full set):`,
              ...score.variant_suggestions
                .slice(0, 3)
                .map((v, i) => `  ${i + 1}. ${v.text}\n     (${v.why})`),
            ].join('\n'),
          },
        ],
      };
    }

    if (!score && !scoreErrorMsg) {
      // shouldn't happen, but defensive
      return {
        content: [{ type: 'text', text: 'Unexpected: scorer returned null without an error.' }],
      };
    }
  }

  // Generate UUID + insert into DB
  const id = randomUUID();
  insertDraft({
    id,
    draft_text: args.draft_text,
    author_wallet: args.author_wallet,
    score,
    references_tx: args.references_tx,
    references_pr: args.references_pr,
  });

  const approvalUrl = `${APPROVAL_BASE_URL}/${id}`;
  const lines = [
    `✓ Draft queued for approval`,
    ``,
    `  ID:       ${id}`,
    `  Status:   pending`,
    ``,
    `  Approval URL:`,
    `    ${approvalUrl}`,
    ``,
    `Next steps:`,
    `  1. Admin opens the approval URL in their wallet-authorised browser`,
    `  2. Admin signs X-Wallet-Auth approval (proves identity)`,
    `  3. Server marks draft 'approved'`,
    `  4. publish_approved tool can then post to X via the configured API token`,
  ];
  if (score) {
    lines.push(``);
    lines.push(`Scoring at queue time:`);
    lines.push(`  Engagement prediction:    ${(score.predicted_engagement * 100).toFixed(0)}%`);
    lines.push(`  Negative signal risk:     ${(score.negative_signal_risk * 100).toFixed(0)}% (within ${NEG_RISK_MAX * 100}% limit)`);
  }
  if (scoreErrorMsg) {
    lines.push(``);
    lines.push(`⚠️  Scoring was skipped: ${scoreErrorMsg}`);
    lines.push(`   AFL-1 §5.3 negative-signal-risk hard limit was NOT enforced.`);
  }
  if (args.references_tx) {
    lines.push(``);
    lines.push(`Referenced tx: ${args.references_tx}`);
    lines.push(`(On-chain attestation will be generated at publish time per AFL-1 §5.5)`);
  }

  return { content: [{ type: 'text', text: lines.join('\n') }] };
}
