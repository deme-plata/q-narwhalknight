// Solution submitter — POSTs found nonces to the Quillon Graph mining endpoint.
//
// The wire format here is a PoC; the production endpoint at
// crates/q-miner/src/solution_submitter.rs:117 (POST /api/v1/mining/submit)
// expects a richer payload including VDF proof fields for Genus-2 mining.
// For SHA3-PoW PoC mode, the server's mining-test endpoint accepts the
// minimal {wallet, nonce, hash, header} payload.

import type { MiningSolution } from './webgpu/runner';

export class Submitter {
  constructor(
    private readonly serverUrl: string,
    private readonly wallet: string,
  ) {}

  async submit(sol: MiningSolution, header: Uint8Array): Promise<{ accepted: boolean; reason?: string }> {
    const body = {
      wallet: this.wallet,
      nonce: sol.nonce,
      hash: bytesToHex(sol.hash),
      header: bytesToHex(header),
      mode: 'sha3-poc', // server-side flag indicating this is the WebGPU-miner PoC mode
    };

    try {
      const res = await fetch(`${this.serverUrl}/api/v1/mining/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      return { accepted: !!data.success, reason: data.error };
    } catch (e: any) {
      return { accepted: false, reason: e.message };
    }
  }
}

function bytesToHex(b: Uint8Array): string {
  return Array.from(b).map((x) => x.toString(16).padStart(2, '0')).join('');
}
