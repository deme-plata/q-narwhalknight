//! Shamir Secret Sharing implementation
//!
//! Provides (k,n) threshold secret sharing over finite fields.

pub mod field;
pub mod polynomial;
pub mod sharing;

pub use field::FieldElement256;
pub use polynomial::Polynomial;
pub use sharing::{ShamirShare, shamir_split, shamir_reconstruct};
