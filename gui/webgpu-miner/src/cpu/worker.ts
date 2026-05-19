// CPU mining worker — runs in a Web Worker thread.
// Uses @noble/hashes/sha3 for audited SHA3-256.
//
// Protocol (postMessage):
//   master → worker: { cmd: 'start', headerBytes: Uint8Array, targetBytes: Uint8Array, nonceStart: number, nonceStep: number }
//   master → worker: { cmd: 'stop' }
//   worker → master: { type: 'solution', nonce, hash: Uint8Array }
//   worker → master: { type: 'progress', hashesDone: number }
//
// Each worker takes a stride-based slice of the nonce space (nonceStart, +nonceStep, +2*nonceStep, ...)
// so N workers tile the space without collision.

import { sha3_256 } from '@noble/hashes/sha3';

let running = false;

function le32(buf: Uint8Array, offset: number, value: number) {
  buf[offset] = value & 0xff;
  buf[offset + 1] = (value >>> 8) & 0xff;
  buf[offset + 2] = (value >>> 16) & 0xff;
  buf[offset + 3] = (value >>> 24) & 0xff;
}

function hashLessOrEqual(hash: Uint8Array, target: Uint8Array): boolean {
  // Big-endian comparison from most-significant byte.
  // The hash is interpreted as a big-endian integer; lower = better.
  // Target is also big-endian. Compare byte-by-byte from index 31 down.
  for (let i = 31; i >= 0; i--) {
    if (hash[i] < target[i]) return true;
    if (hash[i] > target[i]) return false;
  }
  return true; // exactly equal counts as a solution
}

self.addEventListener('message', (e: MessageEvent) => {
  const data = e.data as any;
  if (data.cmd === 'start') {
    running = true;
    mine(data.headerBytes, data.targetBytes, data.nonceStart, data.nonceStep);
  } else if (data.cmd === 'stop') {
    running = false;
  }
});

async function mine(header: Uint8Array, target: Uint8Array, start: number, step: number) {
  // Pre-allocate the message buffer: header (32) || nonce (4) = 36 bytes
  const msg = new Uint8Array(36);
  msg.set(header, 0);

  let nonce = start;
  let hashesDone = 0;
  let lastProgress = performance.now();

  while (running) {
    le32(msg, 32, nonce);
    const hash = sha3_256(msg);
    if (hashLessOrEqual(hash, target)) {
      (self as any).postMessage({ type: 'solution', nonce, hash });
    }
    nonce = (nonce + step) >>> 0; // u32 wrap
    hashesDone++;

    // Report progress every 50ms
    const now = performance.now();
    if (now - lastProgress > 50) {
      (self as any).postMessage({ type: 'progress', hashesDone });
      hashesDone = 0;
      lastProgress = now;
      // Yield to allow stop signal to be received
      await new Promise((r) => setTimeout(r, 0));
    }
  }
}
