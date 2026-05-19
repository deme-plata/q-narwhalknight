#!/usr/bin/env node
// Quillon Twitter MCP server — entry point.
//
// Exposes 4 tools (Layers 1-3 of the spec at
// docs/twitter-mcp-with-x-algorithm-spec.md):
//
//   draft_tweet         — score a draft + return variant suggestions
//   score_draft         — score arbitrary text (no DB write)
//   queue_for_approval  — persist a draft + return admin approval URL
//   list_my_drafts      — read-only list of drafts (by author or pending)
//
// Layers 4-5 (publish_approved, on-chain attestation) require:
//   - X API OAuth2 setup (separate concern, requires account holder action)
//   - PR #87 (Agent Fiber Lane) merged + admin wallet signature flow live
// They land in a follow-up commit.

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';

import { scoreDraftHandler, scoreDraftSchema } from './tools/score_draft.js';
import { draftTweetHandler, draftTweetSchema } from './tools/draft_tweet.js';
import {
  queueForApprovalHandler,
  queueForApprovalSchema,
} from './tools/queue_for_approval.js';
import { listMyDraftsHandler, listMyDraftsSchema } from './tools/list_my_drafts.js';
import { scorerHealth, scorerVersion } from './scorer_client.js';

// ─── Tool registry ──────────────────────────────────────────────────────

const TOOLS = {
  draft_tweet: {
    description:
      'Score a tweet draft against xAI x-algorithm (engagement + negative-signal risk + variant suggestions). Returns refusal if neg-signal-risk > 0.15 hard limit per AFL-1 §5.3.',
    schema: draftTweetSchema,
    handler: draftTweetHandler,
  },
  score_draft: {
    description:
      'Score arbitrary tweet text (no DB write, no side effects). Returns per-action probabilities + suggestions. Use this to iterate on phrasings before queueing.',
    schema: scoreDraftSchema,
    handler: scoreDraftHandler,
  },
  queue_for_approval: {
    description:
      'Persist a tweet draft to the SQLite queue and return an admin-approval URL. The admin must visit the URL and sign an X-Wallet-Auth challenge to approve. Hard-refuses drafts with neg-signal-risk > 0.15.',
    schema: queueForApprovalSchema,
    handler: queueForApprovalHandler,
  },
  list_my_drafts: {
    description:
      "List drafts for an author wallet (or all pending drafts if no wallet specified — admin view).",
    schema: listMyDraftsSchema,
    handler: listMyDraftsHandler,
  },
} as const;

// ─── Server setup ───────────────────────────────────────────────────────

const server = new Server(
  { name: 'quillon-twitter-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: Object.entries(TOOLS).map(([name, t]) => ({
    name,
    description: t.description,
    inputSchema: zodSchemaToJsonSchema(t.schema),
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const name = req.params.name as keyof typeof TOOLS;
  const tool = TOOLS[name];
  if (!tool) {
    throw new Error(`Unknown tool: ${name}`);
  }
  return await tool.handler(req.params.arguments as any);
});

// ─── Helpers ────────────────────────────────────────────────────────────

function zodSchemaToJsonSchema(schemaObj: Record<string, z.ZodTypeAny>) {
  const properties: Record<string, any> = {};
  const required: string[] = [];

  for (const [key, zodType] of Object.entries(schemaObj)) {
    const description = (zodType._def as any).description;
    const optional = zodType.isOptional();
    let typeStr = 'string';
    let inner = zodType;
    // Unwrap optionals
    while ((inner as any)._def?.typeName === 'ZodOptional') {
      inner = (inner as any)._def.innerType;
    }
    const typeName = (inner._def as any).typeName;
    if (typeName === 'ZodNumber') typeStr = 'number';
    else if (typeName === 'ZodBoolean') typeStr = 'boolean';
    else if (typeName === 'ZodArray') {
      properties[key] = { type: 'array', items: { type: 'string' }, description };
      if (!optional) required.push(key);
      continue;
    }
    properties[key] = { type: typeStr, description };
    if (!optional) required.push(key);
  }
  return { type: 'object', properties, required };
}

// ─── Start ──────────────────────────────────────────────────────────────

async function main() {
  // Quick startup health check — log a warning if the scorer sidecar
  // isn't running yet, but don't block startup. The scorer might come
  // up later, and read-only tools (list_my_drafts) work regardless.
  const healthy = await scorerHealth();
  if (!healthy) {
    console.error(
      '[twitter-mcp] WARNING: x-algorithm-scorer sidecar not reachable at startup. ' +
        'Tools that need scoring (draft_tweet, queue_for_approval) will return errors until it comes up. ' +
        'Start with: cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer && cargo run --release',
    );
  } else {
    const v = await scorerVersion();
    console.error(
      `[twitter-mcp] x-algorithm-scorer reachable: ${v?.model ?? 'unknown'} (Layer ${v?.layer ?? '?'})`,
    );
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('[twitter-mcp] MCP server listening on stdio');
}

main().catch((e) => {
  console.error('[twitter-mcp] fatal:', e);
  process.exit(1);
});
