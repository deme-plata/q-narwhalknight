// list_my_drafts MCP tool — read-only query of drafts for a given
// author wallet OR of all pending drafts (admin view).

import { z } from 'zod';
import { listMyDrafts, listPending } from '../db.js';
import type { QueuedDraft } from '../types.js';

export const listMyDraftsSchema = {
  author_wallet: z
    .string()
    .regex(/^qnk[0-9a-f]{64}$/)
    .optional()
    .describe('qnk... address. Omit for admin view (returns all pending drafts).'),
  limit: z
    .number()
    .int()
    .positive()
    .max(100)
    .optional()
    .describe('Max number of drafts to return (default 25, max 100)'),
};

export async function listMyDraftsHandler(args: {
  author_wallet?: string;
  limit?: number;
}): Promise<{ content: Array<{ type: 'text'; text: string }> }> {
  const limit = args.limit ?? 25;
  const drafts: QueuedDraft[] = args.author_wallet
    ? listMyDrafts(args.author_wallet, limit)
    : listPending(limit);

  if (drafts.length === 0) {
    return {
      content: [
        {
          type: 'text',
          text: args.author_wallet
            ? `No drafts found for ${args.author_wallet.slice(0, 12)}...`
            : 'No pending drafts.',
        },
      ],
    };
  }

  const lines = [
    args.author_wallet
      ? `=== Drafts for ${args.author_wallet.slice(0, 12)}... (${drafts.length}) ===`
      : `=== Pending drafts (${drafts.length}) ===`,
    '',
  ];

  for (const d of drafts) {
    lines.push(`[${d.status.toUpperCase()}] ${d.id}`);
    lines.push(`  ${d.draft_text.slice(0, 140)}${d.draft_text.length > 140 ? '…' : ''}`);
    lines.push(`  created: ${d.created_at}`);
    if (d.score) {
      lines.push(
        `  scored:  engagement=${(d.score.predicted_engagement * 100).toFixed(0)}%, neg-risk=${(d.score.negative_signal_risk * 100).toFixed(0)}%`,
      );
    }
    if (d.status === 'approved') {
      lines.push(`  approved: by ${d.approved_by?.slice(0, 12)}... at ${d.approved_at}`);
    }
    if (d.references_tx) lines.push(`  references tx: ${d.references_tx.slice(0, 16)}...`);
    if (d.references_pr) lines.push(`  references PR: #${d.references_pr}`);
    lines.push('');
  }

  return { content: [{ type: 'text', text: lines.join('\n') }] };
}
