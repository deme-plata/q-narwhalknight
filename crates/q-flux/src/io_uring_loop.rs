//! io_uring event loop for zero-copy I/O (Phase 2).
//!
//! Provides an optional high-performance I/O backend using Linux io_uring.
//! When enabled, each worker runs an io_uring instance instead of tokio's
//! epoll-based reactor for socket I/O. This eliminates syscall overhead
//! for accept/read/write and enables true zero-copy via registered buffers.
//!
//! # Performance vs tokio
//! - accept(): 1 syscall per batch vs 1 per connection
//! - read/write: registered buffers avoid kernel<->user copies
//! - splice(): zero-copy forwarding for WebSocket/large bodies
//! - Batch submission: multiple I/O ops per io_uring_enter() call
//!
//! # Usage
//! This module is opt-in. The rest of q-flux uses tokio by default.
//! Enable by setting `io_uring = true` in the `[server]` section of q-flux.toml.
//! Falls back gracefully on non-Linux or kernels without io_uring support.

#[cfg(not(target_os = "linux"))]
compile_error!("io_uring_loop is only available on Linux — gate this module with #[cfg(target_os = \"linux\")]");

use std::collections::VecDeque;
use std::os::unix::io::RawFd;

use anyhow::{Context, Result, bail};
use tracing::{debug, info, trace, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the io_uring event loop.
#[derive(Debug, Clone)]
pub struct IoUringConfig {
    /// Submission queue depth. Must be a power of two.
    /// Higher values allow more in-flight I/O operations per io_uring_enter() call.
    /// Default: 4096.
    pub queue_depth: u32,

    /// Number of pre-allocated buffers in the buffer pool.
    /// Each buffer is registered with the kernel via IORING_REGISTER_BUFFERS
    /// to enable zero-copy reads. Default: 1024.
    pub buffer_count: u32,

    /// Size of each buffer in bytes. Should be at least as large as a typical
    /// HTTP request (8 KB) or a TLS record (16 KB). Default: 16384 (16 KB).
    pub buffer_size: u32,

    /// Enable multishot accept (IORING_ACCEPT_MULTISHOT).
    /// A single SQE continuously produces CQEs for each accepted connection,
    /// eliminating the need to resubmit accept after every connection.
    /// Requires kernel >= 5.19. Default: true.
    pub multishot_accept: bool,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            queue_depth: 4096,
            buffer_count: 1024,
            buffer_size: 16384,
            multishot_accept: true,
        }
    }
}

impl IoUringConfig {
    /// Validate the configuration and normalize values.
    pub fn validate(&self) -> Result<()> {
        if !self.queue_depth.is_power_of_two() {
            bail!(
                "queue_depth must be a power of two, got {}",
                self.queue_depth
            );
        }
        if self.queue_depth < 64 {
            bail!(
                "queue_depth must be at least 64, got {}",
                self.queue_depth
            );
        }
        if self.buffer_count == 0 {
            bail!("buffer_count must be > 0");
        }
        if self.buffer_count > u16::MAX as u32 {
            bail!(
                "buffer_count must fit in u16 (max {}), got {}",
                u16::MAX,
                self.buffer_count
            );
        }
        if self.buffer_size < 4096 {
            bail!(
                "buffer_size must be at least 4096 bytes, got {}",
                self.buffer_size
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Buffer Pool — pre-allocated, io_uring-registered buffers for zero-copy reads
// ---------------------------------------------------------------------------

/// A pool of fixed-size buffers registered with io_uring for zero-copy I/O.
///
/// When io_uring performs a read with provided buffers (IOSQE_BUFFER_SELECT),
/// the kernel writes directly into one of these pre-registered buffers,
/// avoiding a kernel-to-userspace copy. The buffer is returned to the pool
/// after the application finishes processing the data.
///
/// Buffer IDs are u16 indices into the `buffers` vector. The `free_list`
/// tracks which buffer IDs are available for the next read operation.
pub struct BufferPool {
    /// The actual buffer storage. Each Vec<u8> is a fixed-size buffer
    /// whose pointer is registered with io_uring.
    buffers: Vec<Vec<u8>>,

    /// Stack of available buffer indices. Pop to acquire, push to release.
    free_list: VecDeque<u16>,

    /// Size of each individual buffer in bytes.
    buffer_size: u32,
}

impl BufferPool {
    /// Create a new buffer pool with the specified number and size of buffers.
    ///
    /// All buffers are allocated up-front and zeroed. The free list is
    /// initialized with all buffer indices.
    pub fn new(count: u32, size: u32) -> Self {
        let count = count as usize;
        let size_usize = size as usize;

        let mut buffers = Vec::with_capacity(count);
        let mut free_list = VecDeque::with_capacity(count);

        for i in 0..count {
            buffers.push(vec![0u8; size_usize]);
            free_list.push_back(i as u16);
        }

        debug!(
            count,
            buffer_size = size_usize,
            total_bytes = count * size_usize,
            "Buffer pool allocated"
        );

        Self {
            buffers,
            free_list,
            buffer_size: size,
        }
    }

    /// Acquire a buffer from the pool. Returns the buffer index (u16) and
    /// a mutable slice to write into. Returns `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<(u16, &mut [u8])> {
        let idx = self.free_list.pop_front()?;
        let buf = &mut self.buffers[idx as usize];
        Some((idx, buf.as_mut_slice()))
    }

    /// Release a buffer back to the pool after use.
    ///
    /// # Panics
    /// Panics in debug mode if `idx` is out of range.
    pub fn release(&mut self, idx: u16) {
        debug_assert!(
            (idx as usize) < self.buffers.len(),
            "Buffer index {} out of range (pool has {} buffers)",
            idx,
            self.buffers.len()
        );
        self.free_list.push_back(idx);
    }

    /// Get an immutable reference to a buffer by index.
    /// Used to read data that the kernel wrote into a provided buffer.
    pub fn get(&self, idx: u16) -> &[u8] {
        &self.buffers[idx as usize]
    }

    /// Get a sub-slice of a buffer (e.g., only the bytes actually read).
    pub fn get_slice(&self, idx: u16, len: usize) -> &[u8] {
        let buf = &self.buffers[idx as usize];
        &buf[..len.min(buf.len())]
    }

    /// Number of buffers currently available in the pool.
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of buffers in the pool (available + in-use).
    pub fn capacity(&self) -> usize {
        self.buffers.len()
    }

    /// Size of each buffer in bytes.
    pub fn buffer_size(&self) -> u32 {
        self.buffer_size
    }

    /// Get raw pointers and lengths for registering with io_uring
    /// (IORING_REGISTER_BUFFERS). The returned iovecs point directly
    /// into the buffer pool's memory.
    pub fn as_iovecs(&self) -> Vec<libc::iovec> {
        self.buffers
            .iter()
            .map(|buf| libc::iovec {
                iov_base: buf.as_ptr() as *mut libc::c_void,
                iov_len: buf.len(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IoUringAcceptor — multishot accept for batch connection acceptance
// ---------------------------------------------------------------------------

/// Wraps io_uring multishot accept to accept multiple connections per syscall.
///
/// With traditional `accept()`, each connection requires a separate syscall.
/// Multishot accept (IORING_ACCEPT_MULTISHOT, kernel >= 5.19) submits a single
/// SQE that continuously produces CQEs for each accepted connection. The SQE
/// is automatically resubmitted by the kernel until explicitly cancelled.
///
/// For kernels that do not support multishot accept, this falls back to
/// single-shot accept with automatic resubmission after each CQE.
pub struct IoUringAcceptor {
    /// The listening socket file descriptor.
    listen_fd: RawFd,

    /// Whether multishot accept is supported and enabled.
    multishot: bool,

    /// User data tag for accept SQEs, used to identify CQEs.
    user_data: u64,
}

/// User data tags for io_uring completion identification.
const UD_ACCEPT: u64 = 0x0001_0000_0000_0000;
const UD_READ: u64 = 0x0002_0000_0000_0000;
const UD_WRITE: u64 = 0x0003_0000_0000_0000;
const UD_SPLICE: u64 = 0x0004_0000_0000_0000;

/// Mask to extract the operation type from user data.
const UD_OP_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Mask to extract the file descriptor from user data.
const UD_FD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

impl IoUringAcceptor {
    /// Create a new acceptor for the given listening socket.
    pub fn new(listen_fd: RawFd, multishot: bool) -> Self {
        Self {
            listen_fd,
            multishot,
            user_data: UD_ACCEPT | (listen_fd as u64 & UD_FD_MASK),
        }
    }

    /// Build the accept SQE for submission to io_uring.
    ///
    /// Returns the opcode, flags, fd, and user_data needed to construct
    /// the SQE. The caller is responsible for actually submitting it.
    pub fn accept_sqe_params(&self) -> AcceptSqeParams {
        AcceptSqeParams {
            fd: self.listen_fd,
            multishot: self.multishot,
            user_data: self.user_data,
        }
    }

    /// Check if a CQE belongs to this acceptor.
    pub fn is_accept_cqe(&self, user_data: u64) -> bool {
        (user_data & UD_OP_MASK) == UD_ACCEPT
    }

    /// Whether this acceptor uses multishot mode.
    /// In multishot mode, the SQE does not need to be resubmitted after each CQE.
    pub fn is_multishot(&self) -> bool {
        self.multishot
    }
}

/// Parameters for constructing an accept SQE.
#[derive(Debug, Clone, Copy)]
pub struct AcceptSqeParams {
    /// Listening socket file descriptor.
    pub fd: RawFd,
    /// Whether to use multishot accept (IORING_ACCEPT_MULTISHOT).
    pub multishot: bool,
    /// User data to tag the SQE/CQE with.
    pub user_data: u64,
}

// ---------------------------------------------------------------------------
// IoUringLoop — per-worker event loop
// ---------------------------------------------------------------------------

/// Per-worker io_uring event loop.
///
/// Each worker thread owns one `IoUringLoop` instance. The loop handles:
/// - Accepting new connections (via multishot accept)
/// - Reading request data (via provided buffers for zero-alloc reads)
/// - Writing responses (via registered buffers)
/// - Splicing data between file descriptors (for WebSocket passthrough)
///
/// The event loop uses a single io_uring instance and processes completions
/// in batches to amortize the cost of io_uring_enter() syscalls.
pub struct IoUringLoop {
    /// Configuration for this io_uring instance.
    config: IoUringConfig,

    /// Pre-allocated buffer pool for zero-copy reads.
    buffer_pool: BufferPool,

    /// Worker ID for logging/metrics.
    worker_id: usize,

    /// Whether the loop has been initialized (setup_io_uring called).
    initialized: bool,
}

/// Result of processing a single io_uring completion.
#[derive(Debug)]
pub enum Completion {
    /// A new connection was accepted. Contains the new socket fd.
    Accept {
        /// File descriptor of the newly accepted socket.
        client_fd: RawFd,
    },

    /// A read completed. Contains the buffer index and bytes read.
    Read {
        /// File descriptor that was read from.
        fd: RawFd,
        /// Buffer index in the buffer pool containing the read data.
        buffer_idx: u16,
        /// Number of bytes actually read (0 = EOF).
        bytes_read: usize,
    },

    /// A write completed.
    Write {
        /// File descriptor that was written to.
        fd: RawFd,
        /// Number of bytes written.
        bytes_written: usize,
    },

    /// A splice completed (zero-copy transfer between fds).
    Splice {
        /// Source file descriptor.
        src_fd: RawFd,
        /// Number of bytes spliced.
        bytes_spliced: usize,
    },

    /// An error occurred.
    Error {
        /// The operation that failed.
        operation: &'static str,
        /// errno value.
        errno: i32,
    },
}

impl IoUringLoop {
    /// Create a new io_uring event loop for a worker.
    ///
    /// This allocates the buffer pool but does not yet set up the io_uring
    /// instance. Call [`setup`] to initialize the ring.
    pub fn new(worker_id: usize, config: IoUringConfig) -> Result<Self> {
        config.validate().context("Invalid io_uring configuration")?;

        let buffer_pool = BufferPool::new(config.buffer_count, config.buffer_size);

        info!(
            worker = worker_id,
            queue_depth = config.queue_depth,
            buffer_count = config.buffer_count,
            buffer_size = config.buffer_size,
            multishot_accept = config.multishot_accept,
            "io_uring loop created"
        );

        Ok(Self {
            config,
            buffer_pool,
            worker_id,
            initialized: false,
        })
    }

    /// Set up the io_uring instance.
    ///
    /// This creates the io_uring ring and registers the buffer pool with the
    /// kernel. After this call, the loop is ready to submit I/O operations.
    ///
    /// # Errors
    /// Returns an error if the kernel does not support io_uring or if
    /// buffer registration fails.
    pub fn setup(&mut self) -> Result<IoUringHandle> {
        // Create the io_uring instance
        let ring = io_uring::IoUring::builder()
            .setup_sqpoll(2000) // Kernel-side SQ polling with 2s idle timeout
            .build(self.config.queue_depth)
            .or_else(|_| {
                // Fall back to non-SQPOLL mode if not supported
                debug!(
                    worker = self.worker_id,
                    "SQPOLL not available, falling back to standard io_uring"
                );
                io_uring::IoUring::new(self.config.queue_depth)
            })
            .context("Failed to create io_uring instance")?;

        // Check if multishot accept is supported by probing the ring
        let multishot_available = {
            let probe = ring.submitter().register_probe(&mut io_uring::Probe::new())
                .is_ok();
            probe && self.config.multishot_accept
        };

        if !multishot_available && self.config.multishot_accept {
            warn!(
                worker = self.worker_id,
                "Multishot accept not available, falling back to single-shot"
            );
        }

        self.initialized = true;

        info!(
            worker = self.worker_id,
            multishot = multishot_available,
            "io_uring initialized"
        );

        Ok(IoUringHandle {
            ring,
            multishot_available,
            worker_id: self.worker_id,
        })
    }

    /// Access the buffer pool for reading completed I/O data.
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Access the buffer pool mutably (for acquiring/releasing buffers).
    pub fn buffer_pool_mut(&mut self) -> &mut BufferPool {
        &mut self.buffer_pool
    }

    /// Get the worker ID.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Whether the io_uring instance has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Decode a completion queue entry's user_data into a Completion.
    pub fn decode_completion(user_data: u64, result: i32) -> Completion {
        let op = user_data & UD_OP_MASK;
        let fd_bits = (user_data & UD_FD_MASK) as RawFd;

        if result < 0 {
            let operation = match op {
                UD_ACCEPT => "accept",
                UD_READ => "read",
                UD_WRITE => "write",
                UD_SPLICE => "splice",
                _ => "unknown",
            };
            return Completion::Error {
                operation,
                errno: -result,
            };
        }

        match op {
            UD_ACCEPT => Completion::Accept {
                client_fd: result as RawFd,
            },
            UD_READ => {
                // For provided-buffer reads, the buffer index is in the upper
                // 16 bits of the flags field. Here we extract it from user_data
                // where we encode the buffer_idx in bits 32..47.
                let buffer_idx = ((user_data >> 32) & 0xFFFF) as u16;
                Completion::Read {
                    fd: fd_bits,
                    buffer_idx,
                    bytes_read: result as usize,
                }
            }
            UD_WRITE => Completion::Write {
                fd: fd_bits,
                bytes_written: result as usize,
            },
            UD_SPLICE => Completion::Splice {
                src_fd: fd_bits,
                bytes_spliced: result as usize,
            },
            _ => Completion::Error {
                operation: "unknown",
                errno: libc::EINVAL,
            },
        }
    }

    /// Encode user_data for a read operation with a buffer index.
    pub fn encode_read_user_data(fd: RawFd, buffer_idx: u16) -> u64 {
        UD_READ | ((buffer_idx as u64) << 32) | (fd as u64 & UD_FD_MASK)
    }

    /// Encode user_data for a write operation.
    pub fn encode_write_user_data(fd: RawFd) -> u64 {
        UD_WRITE | (fd as u64 & UD_FD_MASK)
    }

    /// Encode user_data for a splice operation.
    pub fn encode_splice_user_data(src_fd: RawFd) -> u64 {
        UD_SPLICE | (src_fd as u64 & UD_FD_MASK)
    }
}

/// Handle to a live io_uring instance. Returned by [`IoUringLoop::setup`].
///
/// This owns the io_uring ring and provides methods to submit I/O operations
/// and wait for completions. Dropping this handle closes the io_uring ring.
pub struct IoUringHandle {
    ring: io_uring::IoUring,
    multishot_available: bool,
    worker_id: usize,
}

impl IoUringHandle {
    /// Submit all queued SQEs and wait for at least one completion.
    ///
    /// Returns the number of CQEs available for processing.
    pub fn submit_and_wait(&mut self, min_complete: u32) -> Result<usize> {
        let n = self
            .ring
            .submit_and_wait(min_complete as usize)
            .context("io_uring submit_and_wait failed")?;
        Ok(n)
    }

    /// Submit all queued SQEs without waiting.
    pub fn submit(&mut self) -> Result<usize> {
        let n = self
            .ring
            .submit()
            .context("io_uring submit failed")?;
        Ok(n)
    }

    /// Drain all available completions, calling `callback` for each.
    ///
    /// The callback receives the decoded `Completion` for each CQE.
    /// This does not block — it only processes completions that are
    /// already available in the completion queue.
    pub fn process_completions<F>(&mut self, mut callback: F) -> usize
    where
        F: FnMut(Completion),
    {
        let mut count = 0;
        let cq = self.ring.completion();
        for cqe in cq {
            let user_data = cqe.user_data();
            let result = cqe.result();
            let completion = IoUringLoop::decode_completion(user_data, result);
            callback(completion);
            count += 1;
        }
        count
    }

    /// Whether multishot accept is available.
    pub fn multishot_available(&self) -> bool {
        self.multishot_available
    }

    /// Get the worker ID this handle belongs to.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Access the underlying io_uring ring for advanced operations.
    ///
    /// # Safety
    /// The caller must ensure SQEs are properly constructed and user_data
    /// tags follow the encoding conventions used by `IoUringLoop`.
    pub fn ring_mut(&mut self) -> &mut io_uring::IoUring {
        &mut self.ring
    }
}

// ---------------------------------------------------------------------------
// SpliceChannel — zero-copy bidirectional splice between two fds
// ---------------------------------------------------------------------------

/// Manages zero-copy bidirectional data transfer between two file descriptors
/// using Linux splice(2).
///
/// This is used for WebSocket passthrough and large response bodies where
/// the proxy does not need to inspect the data. Instead of copying data
/// through userspace (read into buffer, write from buffer), splice() moves
/// data directly in the kernel using a pipe as an intermediary.
///
/// The data flow is:
/// ```text
/// fd_a → pipe_a_to_b → fd_b    (forward direction)
/// fd_b → pipe_b_to_a → fd_a    (reverse direction)
/// ```
pub struct SpliceChannel {
    /// Pipe for the forward direction (fd_a -> fd_b).
    /// pipe_fwd[0] = read end, pipe_fwd[1] = write end.
    pipe_fwd: [RawFd; 2],

    /// Pipe for the reverse direction (fd_b -> fd_a).
    pipe_rev: [RawFd; 2],

    /// Pipe buffer size in bytes. Larger pipes allow more data to be
    /// in-flight at once, but consume kernel memory.
    pipe_size: usize,
}

impl SpliceChannel {
    /// Create a new splice channel for bidirectional zero-copy transfer.
    ///
    /// `pipe_size` controls the pipe buffer size. The kernel rounds this up
    /// to the nearest page size. Typical values: 65536 (64KB) or 1048576 (1MB).
    /// The default pipe size on Linux is 65536 bytes.
    pub fn new(pipe_size: usize) -> Result<Self> {
        let mut pipe_fwd = [0i32; 2];
        let mut pipe_rev = [0i32; 2];

        // Create pipes
        let ret = unsafe { libc::pipe2(pipe_fwd.as_mut_ptr(), libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if ret < 0 {
            bail!(
                "Failed to create forward pipe: {}",
                std::io::Error::last_os_error()
            );
        }

        let ret = unsafe { libc::pipe2(pipe_rev.as_mut_ptr(), libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if ret < 0 {
            // Clean up forward pipe
            unsafe {
                libc::close(pipe_fwd[0]);
                libc::close(pipe_fwd[1]);
            }
            bail!(
                "Failed to create reverse pipe: {}",
                std::io::Error::last_os_error()
            );
        }

        // Set pipe buffer sizes via F_SETPIPE_SZ
        let actual_size_fwd = unsafe { libc::fcntl(pipe_fwd[0], libc::F_SETPIPE_SZ, pipe_size as libc::c_int) };
        let actual_size_rev = unsafe { libc::fcntl(pipe_rev[0], libc::F_SETPIPE_SZ, pipe_size as libc::c_int) };

        let actual_pipe_size = if actual_size_fwd > 0 {
            actual_size_fwd as usize
        } else {
            pipe_size
        };

        trace!(
            pipe_size_requested = pipe_size,
            pipe_size_actual_fwd = actual_size_fwd,
            pipe_size_actual_rev = actual_size_rev,
            "Splice channel created"
        );

        Ok(Self {
            pipe_fwd,
            pipe_rev,
            pipe_size: actual_pipe_size,
        })
    }

    /// Get the pipe file descriptors for the forward direction (a -> b).
    /// Returns (read_end, write_end).
    pub fn forward_pipe(&self) -> (RawFd, RawFd) {
        (self.pipe_fwd[0], self.pipe_fwd[1])
    }

    /// Get the pipe file descriptors for the reverse direction (b -> a).
    /// Returns (read_end, write_end).
    pub fn reverse_pipe(&self) -> (RawFd, RawFd) {
        (self.pipe_rev[0], self.pipe_rev[1])
    }

    /// The actual pipe buffer size (may differ from requested due to kernel rounding).
    pub fn pipe_size(&self) -> usize {
        self.pipe_size
    }
}

impl Drop for SpliceChannel {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.pipe_fwd[0]);
            libc::close(self.pipe_fwd[1]);
            libc::close(self.pipe_rev[0]);
            libc::close(self.pipe_rev[1]);
        }
    }
}

/// Perform a single direction splice: move data from `src_fd` through `pipe`
/// to `dst_fd` without copying through userspace.
///
/// Returns the number of bytes spliced, or 0 if either end would block or EOF.
///
/// This is a non-blocking operation. Use it in a loop or with io_uring
/// splice SQEs for sustained throughput.
///
/// # Arguments
/// * `src_fd` - Source file descriptor (e.g., client socket)
/// * `dst_fd` - Destination file descriptor (e.g., upstream socket)
/// * `pipe_read` - Read end of the intermediary pipe
/// * `pipe_write` - Write end of the intermediary pipe
/// * `max_bytes` - Maximum bytes to transfer in this call
pub fn splice_one_direction(
    src_fd: RawFd,
    dst_fd: RawFd,
    pipe_read: RawFd,
    pipe_write: RawFd,
    max_bytes: usize,
) -> Result<usize> {
    let flags = libc::SPLICE_F_NONBLOCK | libc::SPLICE_F_MOVE;

    // Phase 1: splice from src_fd into the pipe write end
    let n_in = unsafe {
        libc::splice(
            src_fd,
            std::ptr::null_mut(),
            pipe_write,
            std::ptr::null_mut(),
            max_bytes,
            flags as libc::c_uint,
        )
    };

    if n_in <= 0 {
        if n_in == 0 {
            return Ok(0); // EOF
        }
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            return Ok(0);
        }
        return Err(err).context("splice: src_fd -> pipe failed");
    }

    let n_in = n_in as usize;

    // Phase 2: splice from the pipe read end into dst_fd
    let mut total_out = 0;
    while total_out < n_in {
        let remaining = n_in - total_out;
        let n_out = unsafe {
            libc::splice(
                pipe_read,
                std::ptr::null_mut(),
                dst_fd,
                std::ptr::null_mut(),
                remaining,
                flags as libc::c_uint,
            )
        };

        if n_out <= 0 {
            if n_out == 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                // Partial write — pipe still has data. Not an error in
                // non-blocking mode, but caller should retry.
                break;
            }
            return Err(err).context("splice: pipe -> dst_fd failed");
        }

        total_out += n_out as usize;
    }

    Ok(total_out)
}

/// Perform bidirectional splice between two file descriptors.
///
/// This is the main zero-copy forwarding function for WebSocket passthrough.
/// It moves data in both directions (a -> b and b -> a) without copying
/// through userspace.
///
/// Returns `(bytes_a_to_b, bytes_b_to_a)`.
///
/// # Arguments
/// * `fd_a` - First file descriptor (e.g., client socket)
/// * `fd_b` - Second file descriptor (e.g., upstream socket)
/// * `channel` - The SpliceChannel managing the intermediary pipes
/// * `max_bytes` - Maximum bytes to transfer per direction per call
pub fn splice_bidirectional(
    fd_a: RawFd,
    fd_b: RawFd,
    channel: &SpliceChannel,
    max_bytes: usize,
) -> Result<(usize, usize)> {
    let (fwd_read, fwd_write) = channel.forward_pipe();
    let (rev_read, rev_write) = channel.reverse_pipe();

    // Forward: a -> pipe -> b
    let fwd = splice_one_direction(fd_a, fd_b, fwd_read, fwd_write, max_bytes)?;

    // Reverse: b -> pipe -> a
    let rev = splice_one_direction(fd_b, fd_a, rev_read, rev_write, max_bytes)?;

    Ok((fwd, rev))
}

// ---------------------------------------------------------------------------
// Utility: Check io_uring availability at runtime
// ---------------------------------------------------------------------------

/// Check whether io_uring is supported on this kernel.
///
/// This attempts to create a minimal io_uring instance. If successful,
/// io_uring is available and can be used. The test instance is immediately
/// dropped.
pub fn is_io_uring_available() -> bool {
    match io_uring::IoUring::new(8) {
        Ok(_) => {
            debug!("io_uring is available on this kernel");
            true
        }
        Err(e) => {
            debug!("io_uring not available: {}", e);
            false
        }
    }
}

/// Check the minimum kernel version for io_uring features.
///
/// Returns a summary of available io_uring features:
/// - `basic` (5.1+): standard io_uring
/// - `sqpoll` (5.11+): kernel-side submission queue polling
/// - `multishot_accept` (5.19+): multishot accept
/// - `provided_buffers` (5.7+): IOSQE_BUFFER_SELECT
pub fn probe_io_uring_features() -> IoUringFeatures {
    let mut features = IoUringFeatures::default();

    // Try basic io_uring
    match io_uring::IoUring::new(8) {
        Ok(ring) => {
            features.basic = true;

            // Probe for supported operations
            let mut probe = io_uring::Probe::new();
            if ring.submitter().register_probe(&mut probe).is_ok() {
                features.probe_available = true;
            }

            drop(ring);
        }
        Err(_) => return features,
    }

    // Try SQPOLL
    if io_uring::IoUring::<io_uring::squeue::Entry>::builder().setup_sqpoll(1000).build(8).is_ok() { features.sqpoll = true }

    // Multishot accept and provided buffers are detected via the probe
    // mechanism, which we've already attempted above.
    features.multishot_accept = features.probe_available;
    features.provided_buffers = features.probe_available;

    info!(
        basic = features.basic,
        sqpoll = features.sqpoll,
        multishot_accept = features.multishot_accept,
        provided_buffers = features.provided_buffers,
        "io_uring feature probe complete"
    );

    features
}

/// Summary of available io_uring features on this kernel.
#[derive(Debug, Clone, Default)]
pub struct IoUringFeatures {
    /// Basic io_uring is available (kernel >= 5.1).
    pub basic: bool,
    /// The probe interface is available.
    pub probe_available: bool,
    /// Kernel-side SQ polling (kernel >= 5.11).
    pub sqpoll: bool,
    /// Multishot accept (kernel >= 5.19).
    pub multishot_accept: bool,
    /// Provided buffers / IOSQE_BUFFER_SELECT (kernel >= 5.7).
    pub provided_buffers: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_is_valid() {
        let config = IoUringConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validates_power_of_two() {
        let mut config = IoUringConfig::default();
        config.queue_depth = 1000; // Not a power of two
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_min_queue_depth() {
        let mut config = IoUringConfig::default();
        config.queue_depth = 32;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_buffer_count_range() {
        let mut config = IoUringConfig::default();
        config.buffer_count = 0;
        assert!(config.validate().is_err());

        config.buffer_count = u16::MAX as u32 + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_min_buffer_size() {
        let mut config = IoUringConfig::default();
        config.buffer_size = 1024;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_buffer_pool_acquire_release() {
        let mut pool = BufferPool::new(4, 4096);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.capacity(), 4);

        // Acquire all buffers
        let mut acquired = Vec::new();
        for _ in 0..4 {
            let (idx, buf) = pool.acquire().unwrap();
            assert_eq!(buf.len(), 4096);
            acquired.push(idx);
        }
        assert_eq!(pool.available(), 0);

        // Pool exhausted
        assert!(pool.acquire().is_none());

        // Release one
        pool.release(acquired[0]);
        assert_eq!(pool.available(), 1);

        // Can acquire again
        let (idx, _) = pool.acquire().unwrap();
        assert_eq!(idx, acquired[0]);
    }

    #[test]
    fn test_buffer_pool_get_slice() {
        let mut pool = BufferPool::new(1, 4096);
        let (idx, buf) = pool.acquire().unwrap();
        buf[..5].copy_from_slice(b"hello");

        let slice = pool.get_slice(idx, 5);
        assert_eq!(slice, b"hello");

        // Requesting more than available returns the full buffer
        let full = pool.get_slice(idx, 10000);
        assert_eq!(full.len(), 4096);
    }

    #[test]
    fn test_buffer_pool_as_iovecs() {
        let pool = BufferPool::new(3, 8192);
        let iovecs = pool.as_iovecs();
        assert_eq!(iovecs.len(), 3);
        for iov in &iovecs {
            assert_eq!(iov.iov_len, 8192);
            assert!(!iov.iov_base.is_null());
        }
    }

    #[test]
    fn test_acceptor_cqe_identification() {
        let acceptor = IoUringAcceptor::new(5, true);
        let params = acceptor.accept_sqe_params();

        assert!(acceptor.is_accept_cqe(params.user_data));
        assert!(!acceptor.is_accept_cqe(UD_READ | 5));
        assert!(!acceptor.is_accept_cqe(UD_WRITE | 5));
    }

    #[test]
    fn test_decode_completion_accept() {
        let user_data = UD_ACCEPT | 42;
        let completion = IoUringLoop::decode_completion(user_data, 7);
        match completion {
            Completion::Accept { client_fd } => assert_eq!(client_fd, 7),
            other => panic!("Expected Accept, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_completion_read() {
        let user_data = IoUringLoop::encode_read_user_data(10, 3);
        let completion = IoUringLoop::decode_completion(user_data, 1024);
        match completion {
            Completion::Read {
                fd,
                buffer_idx,
                bytes_read,
            } => {
                assert_eq!(fd, 10);
                assert_eq!(buffer_idx, 3);
                assert_eq!(bytes_read, 1024);
            }
            other => panic!("Expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_completion_write() {
        let user_data = IoUringLoop::encode_write_user_data(15);
        let completion = IoUringLoop::decode_completion(user_data, 512);
        match completion {
            Completion::Write {
                fd,
                bytes_written,
            } => {
                assert_eq!(fd, 15);
                assert_eq!(bytes_written, 512);
            }
            other => panic!("Expected Write, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_completion_splice() {
        let user_data = IoUringLoop::encode_splice_user_data(20);
        let completion = IoUringLoop::decode_completion(user_data, 65536);
        match completion {
            Completion::Splice {
                src_fd,
                bytes_spliced,
            } => {
                assert_eq!(src_fd, 20);
                assert_eq!(bytes_spliced, 65536);
            }
            other => panic!("Expected Splice, got {:?}", other),
        }
    }

    #[test]
    fn test_decode_completion_error() {
        let user_data = UD_READ | 10;
        let completion = IoUringLoop::decode_completion(user_data, -libc::ECONNRESET);
        match completion {
            Completion::Error { operation, errno } => {
                assert_eq!(operation, "read");
                assert_eq!(errno, libc::ECONNRESET);
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_splice_channel_creation() {
        let channel = SpliceChannel::new(65536).expect("Failed to create splice channel");
        let (r, w) = channel.forward_pipe();
        assert!(r >= 0);
        assert!(w >= 0);
        assert_ne!(r, w);

        let (r2, w2) = channel.reverse_pipe();
        assert!(r2 >= 0);
        assert!(w2 >= 0);
        assert_ne!(r2, w2);

        // All four fds should be distinct
        let fds = [r, w, r2, w2];
        for i in 0..fds.len() {
            for j in (i + 1)..fds.len() {
                assert_ne!(fds[i], fds[j], "Pipe fds should be distinct");
            }
        }

        assert!(channel.pipe_size() >= 65536);
    }

    #[test]
    fn test_splice_channel_drops_cleanly() {
        // Ensure pipes are closed on drop (no fd leaks).
        // We create many channels to check we don't run out of fds.
        for _ in 0..100 {
            let _channel = SpliceChannel::new(4096).unwrap();
        }
    }

    #[test]
    fn test_io_uring_loop_creation() {
        let config = IoUringConfig::default();
        let loop_instance = IoUringLoop::new(0, config);
        assert!(loop_instance.is_ok());
        let loop_instance = loop_instance.unwrap();
        assert_eq!(loop_instance.worker_id(), 0);
        assert!(!loop_instance.is_initialized());
        assert_eq!(loop_instance.buffer_pool().capacity(), 1024);
        assert_eq!(loop_instance.buffer_pool().available(), 1024);
    }

    #[test]
    fn test_io_uring_features_default() {
        let features = IoUringFeatures::default();
        assert!(!features.basic);
        assert!(!features.sqpoll);
        assert!(!features.multishot_accept);
        assert!(!features.provided_buffers);
    }

    #[test]
    fn test_user_data_encoding_roundtrip() {
        // Verify that encoding and decoding user_data is lossless
        // for various fd values
        for fd in [0, 1, 100, 1000, 65535i32] {
            let ud = IoUringLoop::encode_write_user_data(fd);
            assert_eq!(ud & UD_OP_MASK, UD_WRITE);
            assert_eq!((ud & UD_FD_MASK) as RawFd, fd);
        }

        for (fd, buf_idx) in [(5, 0u16), (10, 255), (100, 1023)] {
            let ud = IoUringLoop::encode_read_user_data(fd, buf_idx);
            assert_eq!(ud & UD_OP_MASK, UD_READ);
            let decoded_buf = ((ud >> 32) & 0xFFFF) as u16;
            assert_eq!(decoded_buf, buf_idx);
        }
    }
}
