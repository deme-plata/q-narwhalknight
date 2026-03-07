//! Lock-free ring buffer implementations using Vyukov's bounded MPMC algorithm.
//!
//! - [`SpscQueue`]: Single-Producer Single-Consumer — lowest latency
//! - [`MpscQueue`]: Multi-Producer Single-Consumer — for fan-in patterns
//!
//! Both use per-slot sequence numbers for coordination:
//! - Slot sequence == producer_pos → slot is writable
//! - Slot sequence == consumer_pos + 1 → slot has data
//!
//! This ensures data is fully written (Release) before the consumer
//! can see it (Acquire), preventing data races without locks.

use crate::slot::Slot;
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── SPSC Queue ───────────────────────────────────────────────────────────

/// Single-Producer Single-Consumer lock-free queue.
///
/// Cache-padded cursors prevent false sharing. No CAS needed on the
/// hot path — just atomic loads and stores.
///
/// # Performance
/// Target: <500 ns round-trip latency on same machine.
pub struct SpscQueue<T> {
    slots: Box<[Slot<T>]>,
    capacity: usize,
    mask: usize,
    producer_pos: CachePadded<AtomicUsize>,
    consumer_pos: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for SpscQueue<T> {}
unsafe impl<T: Send> Sync for SpscQueue<T> {}

impl<T> SpscQueue<T> {
    /// Create a new SPSC queue. Capacity is rounded up to the next power of two.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let mask = capacity - 1;
        let slots: Vec<Slot<T>> = (0..capacity).map(|i| Slot::new(i)).collect();

        Self {
            slots: slots.into_boxed_slice(),
            capacity,
            mask,
            producer_pos: CachePadded::new(AtomicUsize::new(0)),
            consumer_pos: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Try to enqueue a value. Returns `Err(value)` if the queue is full.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        let pos = self.producer_pos.load(Ordering::Relaxed);
        let slot = &self.slots[pos & self.mask];
        let seq = slot.sequence.load(Ordering::Acquire);

        if seq != pos {
            // Slot not ready — queue is full (consumer hasn't caught up)
            return Err(value);
        }

        // Write data, then publish
        unsafe {
            (*slot.data.get()).write(value);
        }
        slot.sequence.store(pos + 1, Ordering::Release);
        self.producer_pos.store(pos + 1, Ordering::Release);
        Ok(())
    }

    /// Try to dequeue a value. Returns `None` if the queue is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let pos = self.consumer_pos.load(Ordering::Relaxed);
        let slot = &self.slots[pos & self.mask];
        let seq = slot.sequence.load(Ordering::Acquire);

        if seq != pos + 1 {
            // Slot not ready — queue is empty (producer hasn't written)
            return None;
        }

        // Read data, then release slot for reuse
        let value = unsafe { (*slot.data.get()).assume_init_read() };
        slot.sequence
            .store(pos + self.capacity, Ordering::Release);
        self.consumer_pos.store(pos + 1, Ordering::Release);
        Some(value)
    }

    /// Approximate number of items in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        let prod = self.producer_pos.load(Ordering::Relaxed);
        let cons = self.consumer_pos.load(Ordering::Relaxed);
        prod.wrapping_sub(cons)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for SpscQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items to drop them
        while self.pop().is_some() {}
    }
}

// ─── MPSC Queue ───────────────────────────────────────────────────────────

/// Multi-Producer Single-Consumer lock-free queue.
///
/// Multiple producers compete via CAS on the producer cursor.
/// A single consumer reads sequentially. Uses Vyukov's bounded MPMC
/// algorithm (with single-consumer simplification).
pub struct MpscQueue<T> {
    slots: Box<[Slot<T>]>,
    capacity: usize,
    mask: usize,
    producer_pos: CachePadded<AtomicUsize>,
    consumer_pos: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for MpscQueue<T> {}
unsafe impl<T: Send> Sync for MpscQueue<T> {}

impl<T> MpscQueue<T> {
    /// Create a new MPSC queue. Capacity is rounded up to the next power of two.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let mask = capacity - 1;
        let slots: Vec<Slot<T>> = (0..capacity).map(|i| Slot::new(i)).collect();

        Self {
            slots: slots.into_boxed_slice(),
            capacity,
            mask,
            producer_pos: CachePadded::new(AtomicUsize::new(0)),
            consumer_pos: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Try to enqueue a value from any producer thread.
    /// Returns `Err(value)` if the queue is full.
    #[inline]
    pub fn push(&self, value: T) -> Result<(), T> {
        loop {
            let pos = self.producer_pos.load(Ordering::Relaxed);
            let slot = &self.slots[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - pos as isize;

            if diff == 0 {
                // Slot available — try to claim this position
                if self
                    .producer_pos
                    .compare_exchange_weak(
                        pos,
                        pos + 1,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // We own this slot — write data and publish
                    unsafe {
                        (*slot.data.get()).write(value);
                    }
                    slot.sequence.store(pos + 1, Ordering::Release);
                    return Ok(());
                }
                // CAS failed — another producer took it, retry
            } else if diff < 0 {
                // Queue full — consumer hasn't caught up
                return Err(value);
            }
            // diff > 0: another producer already advanced past this slot, retry
            std::hint::spin_loop();
        }
    }

    /// Try to dequeue a value. Only one consumer thread should call this.
    /// Returns `None` if the queue is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let pos = self.consumer_pos.load(Ordering::Relaxed);
        let slot = &self.slots[pos & self.mask];
        let seq = slot.sequence.load(Ordering::Acquire);

        if seq != pos + 1 {
            return None;
        }

        let value = unsafe { (*slot.data.get()).assume_init_read() };
        slot.sequence
            .store(pos + self.capacity, Ordering::Release);
        self.consumer_pos.store(pos + 1, Ordering::Release);
        Some(value)
    }

    /// Approximate number of items in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        let prod = self.producer_pos.load(Ordering::Relaxed);
        let cons = self.consumer_pos.load(Ordering::Relaxed);
        prod.wrapping_sub(cons)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for MpscQueue<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ─── SPSC Tests ───

    #[test]
    fn spsc_basic_push_pop() {
        let q = SpscQueue::new(4);
        assert!(q.is_empty());

        q.push(1u64).unwrap();
        q.push(2).unwrap();
        q.push(3).unwrap();
        assert_eq!(q.len(), 3);

        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn spsc_full_queue_rejects() {
        let q = SpscQueue::new(2);
        q.push(10u32).unwrap();
        q.push(20).unwrap();
        assert_eq!(q.push(30), Err(30));
    }

    #[test]
    fn spsc_wrap_around() {
        let q = SpscQueue::new(2);
        for i in 0..100u64 {
            q.push(i).unwrap();
            assert_eq!(q.pop(), Some(i));
        }
    }

    #[test]
    fn spsc_threaded() {
        let q = Arc::new(SpscQueue::new(1024));
        let count = 100_000u64;

        let q_prod = q.clone();
        let producer = thread::spawn(move || {
            for i in 0..count {
                while q_prod.push(i).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        let q_cons = q.clone();
        let consumer = thread::spawn(move || {
            let mut next = 0u64;
            while next < count {
                if let Some(val) = q_cons.pop() {
                    assert_eq!(val, next);
                    next += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    // ─── MPSC Tests ───

    #[test]
    fn mpsc_basic_push_pop() {
        let q = MpscQueue::new(4);
        q.push(1u64).unwrap();
        q.push(2).unwrap();
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn mpsc_multi_producer() {
        let q = Arc::new(MpscQueue::new(4096));
        let per_producer = 10_000u64;
        let num_producers = 4;

        let producers: Vec<_> = (0..num_producers)
            .map(|id| {
                let q = q.clone();
                thread::spawn(move || {
                    for i in 0..per_producer {
                        let val = id * per_producer + i;
                        while q.push(val).is_err() {
                            std::hint::spin_loop();
                        }
                    }
                })
            })
            .collect();

        let total = num_producers * per_producer;
        let consumer = {
            let q = q.clone();
            thread::spawn(move || {
                let mut received = 0u64;
                let mut sum = 0u64;
                while received < total {
                    if let Some(val) = q.pop() {
                        sum += val;
                        received += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                sum
            })
        };

        for p in producers {
            p.join().unwrap();
        }

        let sum = consumer.join().unwrap();
        let expected: u64 = (0..num_producers)
            .flat_map(|id| (0..per_producer).map(move |i| id * per_producer + i))
            .sum();
        assert_eq!(sum, expected);
    }

    #[test]
    fn capacity_is_power_of_two() {
        let q = SpscQueue::<u8>::new(5);
        assert_eq!(q.capacity(), 8);
        let q = SpscQueue::<u8>::new(1);
        assert_eq!(q.capacity(), 2);
        let q = SpscQueue::<u8>::new(1024);
        assert_eq!(q.capacity(), 1024);
    }

    #[test]
    fn drop_cleans_up() {
        use std::sync::atomic::AtomicUsize;
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct DropTracker;
        impl Drop for DropTracker {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let q = SpscQueue::new(8);
            q.push(DropTracker).unwrap();
            q.push(DropTracker).unwrap();
            q.push(DropTracker).unwrap();
            // Drop queue with 3 items inside
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }
}
