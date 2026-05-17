// TURN allocation state (RFC 5766 §5–6).
//
// Each allocation has:
//   - A relay UDP socket bound to an ephemeral port on the server's public IP
//   - A set of permitted peer IP addresses (CreatePermission)
//   - A set of channel bindings (ChannelBind): channel number ↔ peer SocketAddr
//   - An expiry instant

use dashmap::DashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;

const PERMISSION_TTL: Duration = Duration::from_secs(300);   // RFC 5766 §8
const CHANNEL_TTL:    Duration = Duration::from_secs(600);   // RFC 5766 §11

pub struct Permission {
    pub peer_ip:  IpAddr,
    pub expires:  Instant,
}

pub struct Channel {
    pub number:  u16,
    pub peer:    SocketAddr,
    pub expires: Instant,
}

pub struct Allocation {
    pub relay_socket: Arc<UdpSocket>,
    pub relay_addr:   SocketAddr,
    pub client_addr:  SocketAddr,
    expires:          parking_lot::Mutex<Instant>,
    /// username that created this allocation (for refresh credential checks)
    pub username:     String,
    permissions: parking_lot::RwLock<Vec<Permission>>,
    channels:    parking_lot::RwLock<Vec<Channel>>,
}

impl Allocation {
    pub fn new(
        relay_socket: Arc<UdpSocket>,
        relay_addr: SocketAddr,
        client_addr: SocketAddr,
        username: String,
        lifetime: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            relay_socket,
            relay_addr,
            client_addr,
            expires: parking_lot::Mutex::new(Instant::now() + lifetime),
            username,
            permissions: parking_lot::RwLock::new(Vec::new()),
            channels:    parking_lot::RwLock::new(Vec::new()),
        })
    }

    pub fn is_expired(&self) -> bool { Instant::now() >= *self.expires.lock() }

    pub fn refresh(&self, lifetime: Duration) {
        *self.expires.lock() = Instant::now() + lifetime;
    }

    pub fn lifetime_remaining(&self) -> u32 {
        let exp = *self.expires.lock();
        let now = Instant::now();
        if exp <= now { 0 } else { exp.duration_since(now).as_secs() as u32 }
    }

    // ─── Permissions ──────────────────────────────────────────────────────────

    pub fn add_permission(&self, peer_ip: IpAddr) {
        let mut perms = self.permissions.write();
        if let Some(p) = perms.iter_mut().find(|p| p.peer_ip == peer_ip) {
            p.expires = Instant::now() + PERMISSION_TTL;
        } else {
            perms.push(Permission { peer_ip, expires: Instant::now() + PERMISSION_TTL });
        }
        perms.retain(|p| p.expires > Instant::now());
    }

    pub fn has_permission(&self, peer_ip: IpAddr) -> bool {
        let now = Instant::now();
        self.permissions.read().iter().any(|p| p.peer_ip == peer_ip && p.expires > now)
    }

    pub fn permission_count(&self) -> usize {
        let now = Instant::now();
        self.permissions.read().iter().filter(|p| p.expires > now).count()
    }

    // ─── Channels ─────────────────────────────────────────────────────────────

    /// Bind (or refresh) a channel number to a peer address.
    /// Returns false if the channel is already bound to a *different* peer.
    pub fn bind_channel(&self, channel: u16, peer: SocketAddr, max_channels: usize) -> bool {
        let mut chs = self.channels.write();
        chs.retain(|c| c.expires > Instant::now());

        if let Some(c) = chs.iter_mut().find(|c| c.number == channel) {
            if c.peer != peer { return false; }
            c.expires = Instant::now() + CHANNEL_TTL;
            return true;
        }
        if chs.len() >= max_channels { return false; }
        chs.push(Channel { number: channel, peer, expires: Instant::now() + CHANNEL_TTL });
        true
    }

    pub fn channel_for_peer(&self, peer: SocketAddr) -> Option<u16> {
        let now = Instant::now();
        self.channels.read()
            .iter()
            .find(|c| c.peer == peer && c.expires > now)
            .map(|c| c.number)
    }

    pub fn peer_for_channel(&self, channel: u16) -> Option<SocketAddr> {
        let now = Instant::now();
        self.channels.read()
            .iter()
            .find(|c| c.number == channel && c.expires > now)
            .map(|c| c.peer)
    }
}

// ─── Allocation store ─────────────────────────────────────────────────────────

pub struct AllocationStore {
    /// client_addr → allocation
    by_client: DashMap<SocketAddr, Arc<Allocation>>,
    max:       usize,
}

impl AllocationStore {
    pub fn new(max_allocations: usize) -> Arc<Self> {
        Arc::new(Self {
            by_client: DashMap::new(),
            max: max_allocations,
        })
    }

    pub fn insert(&self, alloc: Arc<Allocation>) -> bool {
        self.evict_expired();
        if self.by_client.len() >= self.max { return false; }
        self.by_client.insert(alloc.client_addr, alloc);
        true
    }

    pub fn get(&self, client: SocketAddr) -> Option<Arc<Allocation>> {
        self.by_client.get(&client).map(|r| r.clone())
    }

    pub fn remove(&self, client: SocketAddr) {
        self.by_client.remove(&client);
    }

    pub fn has(&self, client: SocketAddr) -> bool {
        self.by_client.contains_key(&client)
    }

    fn evict_expired(&self) {
        self.by_client.retain(|_, a| !a.is_expired());
    }

    pub fn len(&self) -> usize { self.by_client.len() }
}

// ─── Relay port pool ──────────────────────────────────────────────────────────

/// Tracks which relay ports are currently in use.
pub struct PortPool {
    min:      u16,
    max:      u16,
    next:     parking_lot::Mutex<u16>,
}

impl PortPool {
    pub fn new(min_port: u16, max_port: u16) -> Arc<Self> {
        Arc::new(Self {
            min: min_port,
            max: max_port,
            next: parking_lot::Mutex::new(min_port),
        })
    }

    /// Try to allocate a port, scanning up to (max - min) candidates.
    pub async fn allocate_relay_socket(
        &self,
        public_ip: std::net::IpAddr,
    ) -> Option<(Arc<UdpSocket>, SocketAddr)> {
        let range = (self.max - self.min) as u32;
        let start = {
            let mut n = self.next.lock();
            let s = *n;
            *n = if s >= self.max { self.min } else { s + 1 };
            s
        };

        for i in 0..range {
            let port = self.min + ((start - self.min + i as u16) % (self.max - self.min + 1));
            let bind_addr = SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED), port);
            if let Ok(sock) = UdpSocket::bind(bind_addr).await {
                let relay_addr = SocketAddr::new(public_ip, port);
                return Some((Arc::new(sock), relay_addr));
            }
        }
        None
    }
}
