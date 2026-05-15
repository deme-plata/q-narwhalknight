//! Host-side (off-circuit) helpers for the IVC stack.
//!
//! Each gadget in `crates/q-ivc/src/gadgets/` takes a specific
//! constraint-system allocation shape (PublicKeyVar, SignatureVar, etc.).
//! The chain serializes those values in compact production formats —
//! FIPS-204 bit-packing for Dilithium5 keys/sigs, a custom VDF-NTT
//! witness format for anchor election, BLAKE3-domain-separated bytes
//! for SMT proofs.
//!
//! This module is the bridge. Each submodule provides:
//!   1. A native (off-circuit) struct mirroring the in-circuit allocation.
//!   2. A `from_bytes` constructor that unpacks the production wire
//!      format.
//!   3. An `allocate` method that takes the off-circuit struct and a
//!      `ConstraintSystemRef`, returns the in-circuit allocated form.
//!
//! Splitting the unpacking from the allocation means the bit-fiddly
//! parsing is unit-testable in plain Rust (fast), and the gadget call
//! sites stay terse.

pub mod dilithium_witness;
pub mod anchor_witness;

pub use dilithium_witness::{DilithiumKeyBytes, DilithiumSigBytes};
pub use anchor_witness::AnchorVdfBytes;
