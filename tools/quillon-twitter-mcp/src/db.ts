// SQLite drafts persistence. Drafts live in ~/.quillon-twitter-mcp/drafts.db
// by default; configurable via QTWITTER_DB env var. Lifecycle:
//   pending  → queue_for_approval
//   approved → publish_approved (admin X-Wallet-Auth signed)
//   published → after X API POST returns tweet_id
//   rejected/expired → never publish
//
// Per AFL-1 §5.3 the negative-signal-risk hard limit is enforced BEFORE
// a draft is allowed to queue. See queue_for_approval.ts.

import Database from 'better-sqlite3';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { mkdirSync } from 'node:fs';
import type { DraftStatus, QueuedDraft, ScoreResponse } from './types.js';

const DB_DIR = process.env.QTWITTER_DB_DIR || join(homedir(), '.quillon-twitter-mcp');
const DB_PATH = process.env.QTWITTER_DB || join(DB_DIR, 'drafts.db');

mkdirSync(DB_DIR, { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

db.exec(`
  CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    draft_text TEXT NOT NULL,
    author_wallet TEXT,
    score_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    approved_at TEXT,
    approved_by TEXT,
    approval_signature TEXT,
    references_tx TEXT,
    references_pr INTEGER,
    tweet_id TEXT
  );

  CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts (status);
  CREATE INDEX IF NOT EXISTS idx_drafts_author ON drafts (author_wallet);
  CREATE INDEX IF NOT EXISTS idx_drafts_created ON drafts (created_at);
`);

const insertStmt = db.prepare(`
  INSERT INTO drafts (id, draft_text, author_wallet, score_json, status,
                      created_at, references_tx, references_pr)
  VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
`);

const getStmt = db.prepare(`SELECT * FROM drafts WHERE id = ?`);

const listByAuthorStmt = db.prepare(`
  SELECT * FROM drafts
  WHERE author_wallet = ?
  ORDER BY created_at DESC
  LIMIT ?
`);

const listByStatusStmt = db.prepare(`
  SELECT * FROM drafts
  WHERE status = ?
  ORDER BY created_at ASC
  LIMIT ?
`);

const updateStatusStmt = db.prepare(`
  UPDATE drafts
  SET status = ?, approved_at = ?, approved_by = ?, approval_signature = ?
  WHERE id = ?
`);

const updateTweetIdStmt = db.prepare(`
  UPDATE drafts SET tweet_id = ?, status = 'published' WHERE id = ?
`);

function rowToDraft(row: any): QueuedDraft {
  return {
    id: row.id,
    draft_text: row.draft_text,
    author_wallet: row.author_wallet ?? undefined,
    score: row.score_json ? (JSON.parse(row.score_json) as ScoreResponse) : null,
    status: row.status as DraftStatus,
    created_at: row.created_at,
    approved_at: row.approved_at,
    approved_by: row.approved_by,
    approval_signature: row.approval_signature,
    references_tx: row.references_tx ?? undefined,
    references_pr: row.references_pr ?? undefined,
  };
}

export interface InsertDraftInput {
  id: string;
  draft_text: string;
  author_wallet?: string;
  score: ScoreResponse | null;
  references_tx?: string;
  references_pr?: number;
}

export function insertDraft(input: InsertDraftInput): void {
  insertStmt.run(
    input.id,
    input.draft_text,
    input.author_wallet ?? null,
    input.score ? JSON.stringify(input.score) : null,
    new Date().toISOString(),
    input.references_tx ?? null,
    input.references_pr ?? null,
  );
}

export function getDraft(id: string): QueuedDraft | null {
  const row = getStmt.get(id) as any;
  return row ? rowToDraft(row) : null;
}

export function listMyDrafts(authorWallet: string, limit = 25): QueuedDraft[] {
  return (listByAuthorStmt.all(authorWallet, limit) as any[]).map(rowToDraft);
}

export function listPending(limit = 25): QueuedDraft[] {
  return (listByStatusStmt.all('pending', limit) as any[]).map(rowToDraft);
}

export function approveDraft(
  id: string,
  approvedBy: string,
  approvalSignature: string,
): boolean {
  const result = updateStatusStmt.run(
    'approved',
    new Date().toISOString(),
    approvedBy,
    approvalSignature,
    id,
  );
  return result.changes === 1;
}

export function rejectDraft(id: string): boolean {
  const result = updateStatusStmt.run('rejected', null, null, null, id);
  return result.changes === 1;
}

export function markPublished(id: string, tweetId: string): boolean {
  const result = updateTweetIdStmt.run(tweetId, id);
  return result.changes === 1;
}

export function closeDb(): void {
  db.close();
}
