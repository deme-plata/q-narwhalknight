// HTTP client for the x-algorithm-scorer Rust sidecar.
// Default endpoint: http://127.0.0.1:8090 (localhost-only by default;
// the sidecar binds to 0.0.0.0 only with --bind-all flag).
//
// Configurable via SCORER_URL env var.

import type { ScoreContext, ScoreResponse } from './types.js';

const SCORER_URL = process.env.SCORER_URL || 'http://127.0.0.1:8090';

export class ScorerUnavailableError extends Error {
  constructor(reason: string) {
    super(`x-algorithm-scorer sidecar unreachable: ${reason}. Ensure it's running on ${SCORER_URL} — see tools/quillon-twitter-mcp/crates/x-algorithm-scorer/README.md`);
    this.name = 'ScorerUnavailableError';
  }
}

export async function scoreDraft(
  draftText: string,
  context?: ScoreContext,
): Promise<ScoreResponse> {
  let res: Response;
  try {
    res = await fetch(`${SCORER_URL}/score`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ draft_text: draftText, context }),
    });
  } catch (e: any) {
    throw new ScorerUnavailableError(e?.message ?? 'unknown fetch error');
  }
  if (!res.ok) {
    throw new ScorerUnavailableError(`HTTP ${res.status}`);
  }
  return (await res.json()) as ScoreResponse;
}

export async function scorerHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${SCORER_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function scorerVersion(): Promise<{ layer: number; model: string; description: string } | null> {
  try {
    const res = await fetch(`${SCORER_URL}/version`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
