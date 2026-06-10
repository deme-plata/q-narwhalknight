//! Lock-free slot with sequence-based coordination.
//!
//! Uses the Vyukov bounded MPMC queue algorithm:
//! - Each slot has a `sequence` counter initialized to its index
//! - Producer: sequence == pos → write data → sequence = pos + 1
//! - Consumer: sequence == pos + 1 → read data → sequence = pos + capacity
//!
//! This prevents ABA and ensures data is fully written before
//! the consumer can see it (the sequence store has Release ordering).

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;

/// A single slot in the ring buffer.
#[repr(C)]
pub struct Slot<T> {
    /// Sequence number for coordination. Determines slot state:
    /// - `seq == pos`: slot is available for producer at position `pos`
    /// - `seq == pos + 1`: slot has data ready for consumer at position `pos`
    pub(crate) sequence: AtomicUsize,
    /// The data payload.
    pub(crate) data: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Send> Send for Slot<T> {}
unsafe impl<T: Send> Sync for Slot<T> {}

impl<T> Slot<T> {
    /// Create a new slot with the given initial sequence number.
    /// For a ring buffer of capacity C, slot[i] starts with sequence = i.
    #[inline]
    pub fn new(sequence: usize) -> Self {
        Self {
            sequence: AtomicUsize::new(sequence),
            data: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        // We can't easily tell if data is initialized without extra state.
        // The ring buffer handles cleanup via its own drop impl.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn slot_sequence_init() {
        let slot = Slot::<u64>::new(42);
        assert_eq!(slot.sequence.load(Ordering::Relaxed), 42);
    }
}
