//! q-ivc — Incrementally Verifiable Computation circuits for Q-NarwhalKnight.
//!
//! This crate implements the R1CS circuit gadgets and top-level composition
//! circuit for the recursive SNARK described in:
//!   papers/RECURSIVE_SNARK_WEAK_SUBJECTIVITY_ELIMINATION.md
//!
//! Architecture:
//!   gadgets/  — atomic R1CS sub-circuits (NTT, Poseidon, Dilithium, BLAKE3)
//!   circuits/ — composed circuits (EpochTransitionCircuit)
//!
//! Status: SCAFFOLD — interfaces defined, arithmetic constraints are TODO.
//!
//! Prerequisites before producing valid proofs over real data:
//!   1. State root committed in block headers (P4 consensus change, Q1 2027)
//!   2. NTT gadget: implement polynomial arithmetic in R1CS (ntt.rs TODO)
//!   3. Dilithium verification: implement Az-ct arithmetic (dilithium.rs TODO)
//!   4. Poseidon parameters: fix and coordinate with LatticeGuard prover
//!
//! Integration with existing crates:
//!   q-lattice-guard  — native (non-circuit) NTT, used as reference for gadget
//!   q-zk-snark       — Groth16/PLONK backends this circuit will target
//!   q-recursive-proofs — higher-level recursive proof chain orchestration

pub mod circuits;
pub mod gadgets;
pub mod host;
pub mod recursion;

pub use circuits::EpochTransitionCircuit;
