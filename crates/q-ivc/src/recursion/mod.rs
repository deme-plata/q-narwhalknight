//! Recursive SNARK integration module.
//!
//! This module is the integration surface between the q-ivc circuit gadgets
//! (`gadgets/`) and the higher-level Nova-style folding driver described in
//! the whitepaper:
//!
//!   papers/quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.pdf
//!
//! ## Status
//!
//! Phase 1 (gadgets/EpochTransitionCircuit) — IMPLEMENTED.
//! Phase 2 (Nova recursive folding wrapper)  — NOT YET IMPLEMENTED.
//!
//! The Phase 2 work is tracked on the job board:
//!   docs/deepseek-job-board-nova-phase2-2026-05-14.md  (jobs N1..N8)
//!
//! Today this module contains only the `tip_watcher` integration point: a
//! task that subscribes to the new-block event stream and, when Phase 2
//! lands, will call into `NovaFolder::fold_block`. Until then it counts
//! observed blocks, exposes a stub `fold_block` that returns a clear "not
//! yet" error, and logs at TRACE every 100 blocks so the integration is
//! observable without being noisy.
//!
//! ## Wiring Phase 2
//!
//! When jobs N1..N3 land (NovaFolder struct + fold_block method + upgrade
//! gate), the only change needed in this module is the body of
//! `TipWatcher::fold_block` — see the marker comment in
//! `tip_watcher.rs`.

pub mod tip_watcher;
pub mod step_circuit;

pub use tip_watcher::{TipWatcher, TipWatcherState};
pub use step_circuit::{
    DeltaStepCircuit, FoldError, StepCircuitAdapter, StepIO, STEP_Z_LEN,
    fold_native,
};
