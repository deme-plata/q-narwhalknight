// WebGPU miner runner — dispatches the Keccak shader, reads back solutions.
//
// Usage:
//   const miner = await WebGpuMiner.create();
//   miner.onSolution = (nonce, hash) => console.log(nonce, hash);
//   await miner.start(headerBytes, targetBytes);
//   // ...
//   miner.stop();
//
// Dispatches one batch (~262144 nonces) per requestAnimationFrame tick.
// Reports hashrate via miner.hashrate (H/s, rolling 5-second window).

const WORKGROUP_SIZE = 256;
const WORKGROUPS_PER_DISPATCH = 1024;
const NONCES_PER_DISPATCH = WORKGROUP_SIZE * WORKGROUPS_PER_DISPATCH;
const MAX_RESULTS_PER_DISPATCH = 256;

export interface MiningSolution {
  nonce: number;
  hash: Uint8Array; // 32 bytes
}

export class WebGpuMiner {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private inputBuffer: GPUBuffer;
  private resultCountBuffer: GPUBuffer;
  private resultsBuffer: GPUBuffer;
  private resultsReadBuffer: GPUBuffer;
  private resultCountReadBuffer: GPUBuffer;
  private bindGroup: GPUBindGroup;

  private nonceBase = 0;
  private running = false;
  private headerBytes: Uint8Array = new Uint8Array(32);
  private targetBytes: Uint8Array = new Uint8Array(32);

  private hashesInWindow: { t: number; n: number }[] = [];
  public onSolution: (sol: MiningSolution) => void = () => {};
  public onHashrate: (hashesPerSec: number) => void = () => {};

  static async create(shaderCode: string): Promise<WebGpuMiner> {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available (need Chrome 113+ / Edge / Safari TP)');
    }
    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter available');
    const device = await adapter.requestDevice();
    return new WebGpuMiner(device, shaderCode);
  }

  private constructor(device: GPUDevice, shaderCode: string) {
    this.device = device;

    const module = device.createShaderModule({ code: shaderCode });
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'mine' },
    });

    // Input: 32-byte header + 32-byte target + 4-byte nonce_base + 12-byte pad = 80 bytes,
    // aligned to 16 in WGSL → padded to 96 (header[8] + target[8] + nonce_base + 3 pad).
    this.inputBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.resultCountBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Each Solution: u32 nonce + 8×u32 hash = 36 bytes, padded to 48 for alignment.
    this.resultsBuffer = device.createBuffer({
      size: 48 * MAX_RESULTS_PER_DISPATCH,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.resultsReadBuffer = device.createBuffer({
      size: 48 * MAX_RESULTS_PER_DISPATCH,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.resultCountReadBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.resultCountBuffer } },
        { binding: 2, resource: { buffer: this.resultsBuffer } },
      ],
    });
  }

  async start(headerBytes: Uint8Array, targetBytes: Uint8Array) {
    if (headerBytes.length !== 32) throw new Error('header must be 32 bytes');
    if (targetBytes.length !== 32) throw new Error('target must be 32 bytes');
    this.headerBytes = headerBytes;
    this.targetBytes = targetBytes;
    this.nonceBase = 0;
    this.running = true;
    this.loop();
  }

  stop() {
    this.running = false;
  }

  get hashrate(): number {
    if (this.hashesInWindow.length === 0) return 0;
    const now = performance.now();
    const cutoff = now - 5000; // 5s window
    while (this.hashesInWindow.length > 0 && this.hashesInWindow[0].t < cutoff) {
      this.hashesInWindow.shift();
    }
    const total = this.hashesInWindow.reduce((acc, e) => acc + e.n, 0);
    const span = Math.max(now - (this.hashesInWindow[0]?.t ?? now), 1);
    return (total * 1000) / span;
  }

  private async loop() {
    while (this.running) {
      await this.dispatchOnce();
      // Update hashrate window
      this.hashesInWindow.push({ t: performance.now(), n: NONCES_PER_DISPATCH });
      this.onHashrate(this.hashrate);
      // Yield to event loop so UI stays responsive
      await new Promise((r) => requestAnimationFrame(r));
    }
  }

  private async dispatchOnce() {
    // Pack input buffer: header(32B) || target(32B) || nonce_base(4B) || pad(12B) = 80B,
    // but WGSL needs 16-aligned struct → 96B total (uniform).
    const buf = new ArrayBuffer(96);
    const u32 = new Uint32Array(buf);
    const headerView = new Uint32Array(this.headerBytes.buffer, this.headerBytes.byteOffset, 8);
    const targetView = new Uint32Array(this.targetBytes.buffer, this.targetBytes.byteOffset, 8);
    u32.set(headerView, 0); // header[0..8]
    u32.set(targetView, 8); // target[0..8]
    u32[16] = this.nonceBase; // nonce_base
    // u32[17..20] are pad
    this.device.queue.writeBuffer(this.inputBuffer, 0, buf);

    // Reset result counter
    this.device.queue.writeBuffer(this.resultCountBuffer, 0, new Uint32Array([0]));

    // Encode + dispatch
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(WORKGROUPS_PER_DISPATCH);
    pass.end();

    // Copy results out for CPU readback
    encoder.copyBufferToBuffer(this.resultCountBuffer, 0, this.resultCountReadBuffer, 0, 4);
    encoder.copyBufferToBuffer(this.resultsBuffer, 0, this.resultsReadBuffer, 0, 48 * MAX_RESULTS_PER_DISPATCH);

    this.device.queue.submit([encoder.finish()]);

    // Wait for completion + read result count
    await this.resultCountReadBuffer.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(this.resultCountReadBuffer.getMappedRange())[0];
    this.resultCountReadBuffer.unmap();

    if (count > 0) {
      await this.resultsReadBuffer.mapAsync(GPUMapMode.READ);
      const view = new Uint32Array(this.resultsReadBuffer.getMappedRange());
      const numToRead = Math.min(count, MAX_RESULTS_PER_DISPATCH);
      for (let i = 0; i < numToRead; i++) {
        const offset = (i * 48) / 4; // 48 bytes per Solution / 4 bytes per u32
        const nonce = view[offset];
        const hashU32 = view.slice(offset + 1, offset + 9);
        const hashBytes = new Uint8Array(hashU32.buffer.slice(hashU32.byteOffset, hashU32.byteOffset + 32));
        this.onSolution({ nonce, hash: hashBytes });
      }
      this.resultsReadBuffer.unmap();
    }

    this.nonceBase += NONCES_PER_DISPATCH;
  }
}
