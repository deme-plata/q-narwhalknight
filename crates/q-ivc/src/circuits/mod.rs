pub mod epoch_transition;
pub use epoch_transition::{
    EpochTransitionCircuit, EpochTransitionInputs, ValidatorSignatureInput,
    BFT_THRESHOLD, STATE_ROOT_WORDS, VSH_CHUNKS, compute_prior_commitment_native,
};

pub mod delta_block;
pub use delta_block::{
    AnchorWitness, CoinbaseWitness, DeltaBlockCircuit, DeltaBlockInputs,
    TransactionWitness,
};
