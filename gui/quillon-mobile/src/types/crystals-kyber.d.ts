declare module 'crystals-kyber' {
  export function KeyGen1024(): Promise<[Uint8Array, Uint8Array]>;
  export function Encrypt1024(pk: Uint8Array): Promise<[Uint8Array, Uint8Array]>;
  export function Decrypt1024(ct: Uint8Array, sk: Uint8Array): Promise<Uint8Array>;
}
