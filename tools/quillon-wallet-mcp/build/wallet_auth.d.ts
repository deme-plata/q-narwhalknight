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
