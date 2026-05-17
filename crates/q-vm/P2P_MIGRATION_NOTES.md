# P2P Network Migration Notes

The libp2p library has been updated from v0.50.0 to v0.53. This introduces several changes to the API that may require manual adjustments:

## Key Changes

1. **FloodSub to GossipSub**: 
   - Replace `floodsub` with `gossipsub`
   - Replace `Floodsub` with `Gossipsub` 
   - Replace `FloodsubEvent` with `GossipsubEvent`

2. **Network Behavior Changes**:
   - `NetworkBehaviourEventProcess` trait has been deprecated
   - Use `#[behaviour(event_process = true)]` attribute and implement event handling in the struct

3. **Transport Changes**:
   - `mplex` has been largely replaced by `yamux` for multiplexing

4. **Swarm API Changes**:
   - Some swarm methods have changed their signatures
   - `Swarm::dial` usage may need to be updated

## Example Migration:

```rust
// Old code
impl NetworkBehaviourEventProcess<MdnsEvent> for DAGKnightBehaviour {
    fn inject_event(&mut self, event: MdnsEvent) {
        // handle event
    }
}

// New code
#[derive(NetworkBehaviour)]
#[behaviour(event_process = true)]
struct DAGKnightBehaviour {
    gossipsub: Gossipsub,
    mdns: Mdns,
    
    // event handling now happens with methods within the struct impl
}

impl DAGKnightBehaviour {
    fn handle_gossipsub_event(&mut self, event: GossipsubEvent) {
        // handle event
    }
    
    fn handle_mdns_event(&mut self, event: MdnsEvent) {
        // handle event
    }
}
```

For full details on the migration, please refer to the libp2p changelog and documentation.
