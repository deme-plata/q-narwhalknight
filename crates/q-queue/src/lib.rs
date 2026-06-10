//! # q-queue — High-Performance Universal Queue System
//!
//! Lock-free, cache-friendly queues for ultra-low-latency IPC.
//!
//! ## Queue Types
//!
//! - [`SpscQueue`] — Single-Producer Single-Consumer. Lowest latency (<500 ns).
//! - [`MpscQueue`] — Multi-Producer Single-Consumer. Fan-in pattern.
//! - [`Notifier`] — Cross-thread wakeup for blocking consumers.
//!
//! ## Design
//!
//! Uses per-element versioning in a fixed-size ring buffer:
//! - No locks, no mutexes — pure atomic CAS
//! - Cache-padded cursors eliminate false sharing
//! - Power-of-two capacity for branchless modular arithmetic
//! - Version-based ABA prevention without epoch reclamation
//!
//! ## Example
//!
//! ```rust
//! use q_queue::SpscQueue;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let q = Arc::new(SpscQueue::new(1024));
//!
//! let producer = {
//!     let q = q.clone();
//!     thread::spawn(move || {
//!         for i in 0..100u64 {
//!             while q.push(i).is_err() {
//!                 std::hint::spin_loop();
//!             }
//!         }
//!     })
//! };
//!
//! let consumer = {
//!     let q = q.clone();
//!     thread::spawn(move || {
//!         let mut sum = 0u64;
//!         for _ in 0..100 {
//!             loop {
//!                 if let Some(v) = q.pop() {
//!                     sum += v;
//!                     break;
//!                 }
//!                 std::hint::spin_loop();
//!             }
//!         }
//!         sum
//!     })
//! };
//!
//! producer.join().unwrap();
//! assert_eq!(consumer.join().unwrap(), (0..100u64).sum());
//! ```

pub mod slot;
pub mod ring;
pub mod notify;
pub mod persistent;

pub use ring::{SpscQueue, MpscQueue};
pub use notify::Notifier;
pub use persistent::PersistentQueue;
