/**
 * Global type definitions for the Quantum Wallet application
 */

// Extend Navigator interface for WebGPU
interface Navigator {
  gpu?: {
    requestAdapter(options?: any): Promise<any>
  }
}

// Module augmentations for transformers.js
declare module '@xenova/transformers' {
  export function pipeline(
    task: string,
    model?: string,
    options?: any
  ): Promise<any>
}

// NodeJS types for setTimeout return value
declare namespace NodeJS {
  type Timeout = number
}
