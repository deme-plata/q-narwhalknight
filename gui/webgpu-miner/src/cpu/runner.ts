// CPU runner — spawns a pool of Web Workers, aggregates results.
//
// Workers tile the nonce space by stride: worker i covers nonces (i, i+N, i+2N, ...)
// where N = pool size. No coordination overhead, no collision.

import type { MiningSolution } from '../webgpu/runner';

const WORKER_COUNT = Math.max(2, Math.min(navigator.hardwareConcurrency || 4, 16));

export class CpuMiner {
  private workers: Worker[] = [];
  private running = false;
  private hashesInWindow: { t: number; n: number }[] = [];

  public onSolution: (sol: MiningSolution) => void = () => {};
  public onHashrate: (hashesPerSec: number) => void = () => {};

  async start(headerBytes: Uint8Array, targetBytes: Uint8Array) {
    if (headerBytes.length !== 32) throw new Error('header must be 32 bytes');
    if (targetBytes.length !== 32) throw new Error('target must be 32 bytes');
    if (this.running) return;
    this.running = true;
    this.hashesInWindow = [];

    for (let i = 0; i < WORKER_COUNT; i++) {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e: MessageEvent) => this.onWorkerMessage(e);
      worker.postMessage({
        cmd: 'start',
        headerBytes,
        targetBytes,
        nonceStart: i,
        nonceStep: WORKER_COUNT,
      });
      this.workers.push(worker);
    }
  }

  stop() {
    this.running = false;
    for (const w of this.workers) {
      w.postMessage({ cmd: 'stop' });
      w.terminate();
    }
    this.workers = [];
  }

  get workerCount(): number {
    return WORKER_COUNT;
  }

  get hashrate(): number {
    if (this.hashesInWindow.length === 0) return 0;
    const now = performance.now();
    const cutoff = now - 5000;
    while (this.hashesInWindow.length > 0 && this.hashesInWindow[0].t < cutoff) {
      this.hashesInWindow.shift();
    }
    const total = this.hashesInWindow.reduce((acc, e) => acc + e.n, 0);
    const span = Math.max(now - (this.hashesInWindow[0]?.t ?? now), 1);
    return (total * 1000) / span;
  }

  private onWorkerMessage(e: MessageEvent) {
    const data = e.data as any;
    if (data.type === 'solution') {
      this.onSolution({ nonce: data.nonce, hash: data.hash });
    } else if (data.type === 'progress') {
      this.hashesInWindow.push({ t: performance.now(), n: data.hashesDone });
      this.onHashrate(this.hashrate);
    }
  }
}
