// WebGPU miner — main orchestration.
//
// Detects WebGPU availability, falls back to CPU Web Worker pool.
// Wires UI to the runner's hashrate + solution events.

import { WebGpuMiner } from './webgpu/runner';
import { CpuMiner } from './cpu/runner';
import { Submitter } from './submitter';
import keccakShader from './webgpu/keccak.wgsl?raw';

type Miner = WebGpuMiner | CpuMiner;

const $ = (id: string) => document.getElementById(id) as HTMLElement;

let activeMiner: Miner | null = null;
let mode: 'webgpu' | 'cpu' | null = null;
let solutionsFound = 0;
let solutionsAccepted = 0;

// Parse wallet from URL ?wallet=qnk... or localStorage
function getWallet(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get('wallet') || localStorage.getItem('webgpu-miner-wallet') || '';
}

function setStatus(msg: string, level: 'info' | 'warn' | 'error' = 'info') {
  const el = $('status');
  el.textContent = msg;
  el.className = `status status-${level}`;
}

function formatHashrate(hps: number): string {
  if (hps > 1e6) return `${(hps / 1e6).toFixed(2)} MH/s`;
  if (hps > 1e3) return `${(hps / 1e3).toFixed(2)} KH/s`;
  return `${hps.toFixed(1)} H/s`;
}

async function detectMode(): Promise<'webgpu' | 'cpu'> {
  if ('gpu' in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) return 'webgpu';
    } catch (_) {
      // fall through to CPU
    }
  }
  return 'cpu';
}

async function start() {
  const wallet = ($('wallet-input') as HTMLInputElement).value.trim();
  if (!wallet.startsWith('qnk') || wallet.length !== 67) {
    setStatus('Invalid wallet address (must be qnk + 64 hex chars)', 'error');
    return;
  }
  localStorage.setItem('webgpu-miner-wallet', wallet);

  // For PoC mode, use a fixed header (in production: fetch from /api/v1/mining/challenge)
  const header = new Uint8Array(32);
  crypto.getRandomValues(header); // PoC: random per session

  // Easy target for PoC (top 2 bytes must be zero ≈ 1 in 65k hashes finds a solution)
  const target = new Uint8Array(32);
  target.fill(0xff);
  target[31] = 0x00; // most-significant byte
  target[30] = 0x00; // second-most-significant byte

  mode = await detectMode();
  const submitter = new Submitter(window.location.origin, wallet);

  if (mode === 'webgpu') {
    setStatus('Starting WebGPU miner...', 'info');
    const m = await WebGpuMiner.create(keccakShader);
    m.onSolution = (sol) => onSolution(sol, submitter, header);
    m.onHashrate = (hr) => ($('hashrate').textContent = formatHashrate(hr));
    activeMiner = m;
    await m.start(header, target);
    setStatus('WebGPU mining active', 'info');
  } else {
    setStatus('WebGPU not available — falling back to CPU pool', 'warn');
    const m = new CpuMiner();
    m.onSolution = (sol) => onSolution(sol, submitter, header);
    m.onHashrate = (hr) => ($('hashrate').textContent = formatHashrate(hr));
    activeMiner = m;
    await m.start(header, target);
    setStatus(`CPU mining active (${m.workerCount} workers)`, 'info');
  }

  ($('start-btn') as HTMLButtonElement).disabled = true;
  ($('stop-btn') as HTMLButtonElement).disabled = false;
  $('mode-indicator').textContent = mode.toUpperCase();
}

function stop() {
  if (activeMiner) {
    activeMiner.stop();
    activeMiner = null;
  }
  setStatus('Stopped', 'info');
  ($('start-btn') as HTMLButtonElement).disabled = false;
  ($('stop-btn') as HTMLButtonElement).disabled = true;
}

async function onSolution(sol: { nonce: number; hash: Uint8Array }, submitter: Submitter, header: Uint8Array) {
  solutionsFound++;
  $('solutions-found').textContent = String(solutionsFound);
  const result = await submitter.submit(sol, header);
  if (result.accepted) {
    solutionsAccepted++;
    $('solutions-accepted').textContent = String(solutionsAccepted);
  } else {
    // Log but don't error — rejections are common during local dev
    console.warn('Solution rejected:', result.reason);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  ($('wallet-input') as HTMLInputElement).value = getWallet();
  ($('start-btn') as HTMLButtonElement).onclick = start;
  ($('stop-btn') as HTMLButtonElement).onclick = stop;
});
