use std::fmt;
use crate::network::p2p::P2pNetwork;

impl fmt::Debug for P2pNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("P2pNetwork")
            .field("local_peer_id", &self.get_local_peer_id())
            .field("connected_peers_count", &self.get_connected_peers_count())
            .finish()
    }
}
