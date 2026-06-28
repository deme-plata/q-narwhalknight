export declare class SeedNotFoundError extends Error {
    readonly checked: string[];
    constructor(checked: string[]);
}
export declare class SignatureError extends Error {
    readonly address?: string | undefined;
    constructor(message: string, address?: string | undefined);
}
export interface LoadSeedResult {
    seed: string;
    source: string;
}
export declare function loadSeed(opts?: {
    seedArg?: string;
}): LoadSeedResult;
export interface DerivedKeys {
    priv: Uint8Array;
    pub: Uint8Array;
    address: string;
}
export declare function deriveKeys(seed: string): DerivedKeys;
export interface SignedAuth {
    header: string;
    address: string;
    timestamp: number;
    source: string;
}
export declare function signXWalletAuth(reqPath: string, opts?: {
    seedArg?: string;
    timestamp?: number;
}): SignedAuth;
export declare const MIN_TRANSACTION_FEE_BASE = 21000n;
export interface SignedTransferV72 {
    signature: string;
    nonce: number;
    timestamp: number;
    fee: string;
    address: string;
}
/**
 * Build the canonical p2p signable payload for a transfer and Ed25519-sign its
 * SHA3-256 hash using the configured seed. `toAddr` may be qnk-prefixed or raw hex.
 * `amountBase` is the u128 base-24 amount as a bigint (same value posted as `amount`).
 */
export declare function signTransferV72(params: {
    toAddr: string;
    amountBase: bigint;
    nonce: number;
    timestampSecs: number;
    feeBase?: bigint;
    tokenType?: string;
    seedArg?: string;
    dataHex?: string;
}): SignedTransferV72;
