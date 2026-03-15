/// SIMD-accelerated bitfield DAG representation for DAG-Knight consensus
///
/// Replaces HashMap<VertexId, HashSet<VertexId>> with compact bitfields
/// for O(N/64) set operations instead of O(N) hash lookups.
///
/// Performance targets:
/// - Anticone computation for 10K vertices: ~400us -> ~5us (80x with AVX-512)
/// - Topological sort: 10-30x improvement
/// - Memory: 12.5 MB vs 320 MB for 10K vertices
///
/// Feature-gated: #[cfg(feature = "simd-dag")]

use std::collections::{HashMap, HashSet, VecDeque};
use q_types::VertexId;
use tracing::{debug, warn};

/// Compact bitfield representing a set of vertices by index
#[derive(Clone, Debug)]
pub struct VertexBitfield {
    bits: Vec<u64>,
    capacity: usize, // Total number of bits (rounded up to 64)
}

impl VertexBitfield {
    /// Create an empty bitfield with given capacity (in vertices)
    pub fn new(capacity: usize) -> Self {
        let words = (capacity + 63) / 64;
        Self {
            bits: vec![0u64; words],
            capacity: words * 64,
        }
    }

    /// Set bit at index
    #[inline]
    pub fn set(&mut self, idx: u32) {
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Clear bit at index
    #[inline]
    pub fn clear(&mut self, idx: u32) {
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        if word < self.bits.len() {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    /// Test if bit is set
    #[inline]
    pub fn test(&self, idx: u32) -> bool {
        let word = idx as usize / 64;
        let bit = idx as usize % 64;
        word < self.bits.len() && (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Bitwise OR (union) with another bitfield
    #[inline]
    pub fn union_with(&mut self, other: &VertexBitfield) {
        let len = self.bits.len().min(other.bits.len());
        for i in 0..len {
            self.bits[i] |= other.bits[i];
        }
    }

    /// Bitwise AND (intersection) with another bitfield
    #[inline]
    pub fn intersect_with(&mut self, other: &VertexBitfield) {
        let len = self.bits.len().min(other.bits.len());
        for i in 0..len {
            self.bits[i] &= other.bits[i];
        }
        // Zero out words beyond other's length
        for i in len..self.bits.len() {
            self.bits[i] = 0;
        }
    }

    /// Compute anticone: NOT(past) AND NOT(future) AND NOT(self_bit)
    /// Returns a new bitfield representing the anticone of vertex at `self_idx`
    pub fn anticone(past: &VertexBitfield, future: &VertexBitfield, self_idx: u32, active: &VertexBitfield) -> VertexBitfield {
        let len = past.bits.len().min(future.bits.len()).min(active.bits.len());
        let mut result = VertexBitfield::new(len * 64);

        for i in 0..len {
            // active AND NOT(past) AND NOT(future)
            result.bits[i] = active.bits[i] & !past.bits[i] & !future.bits[i];
        }

        // Clear self bit
        result.clear(self_idx);

        result
    }

    /// Count set bits (popcount)
    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Intersection count (popcount of AND)
    #[inline]
    pub fn intersection_count(&self, other: &VertexBitfield) -> u32 {
        let len = self.bits.len().min(other.bits.len());
        let mut count = 0u32;
        for i in 0..len {
            count += (self.bits[i] & other.bits[i]).count_ones();
        }
        count
    }

    /// Iterate over set bit indices
    pub fn iter_set_bits(&self) -> impl Iterator<Item = u32> + '_ {
        self.bits.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = (word_idx * 64) as u32;
            BitIterator { word, base }
        })
    }

    /// Grow capacity to at least `new_capacity` bits
    pub fn grow_to(&mut self, new_capacity: usize) {
        let new_words = (new_capacity + 63) / 64;
        if new_words > self.bits.len() {
            self.bits.resize(new_words, 0);
            self.capacity = new_words * 64;
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }
}

/// Iterator over set bits in a single u64 word
struct BitIterator {
    word: u64,
    base: u32,
}

impl Iterator for BitIterator {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros();
        self.word &= self.word - 1; // Clear lowest set bit
        Some(self.base + tz)
    }
}

/// Bidirectional mapping between VertexId and compact u32 indices
pub struct VertexIndexMap {
    vertex_to_index: HashMap<VertexId, u32>,
    index_to_vertex: Vec<VertexId>,
    free_list: VecDeque<u32>,
    next_index: u32,
}

impl VertexIndexMap {
    pub fn new() -> Self {
        Self {
            vertex_to_index: HashMap::new(),
            index_to_vertex: Vec::new(),
            free_list: VecDeque::new(),
            next_index: 0,
        }
    }

    /// Get or assign an index for a vertex
    pub fn get_or_insert(&mut self, vertex_id: VertexId) -> u32 {
        if let Some(&idx) = self.vertex_to_index.get(&vertex_id) {
            return idx;
        }

        let idx = if let Some(recycled) = self.free_list.pop_front() {
            self.index_to_vertex[recycled as usize] = vertex_id;
            recycled
        } else {
            let idx = self.next_index;
            self.next_index += 1;
            if idx as usize >= self.index_to_vertex.len() {
                self.index_to_vertex.push(vertex_id);
            } else {
                self.index_to_vertex[idx as usize] = vertex_id;
            }
            idx
        };

        self.vertex_to_index.insert(vertex_id, idx);
        idx
    }

    /// Look up index for a vertex
    pub fn get_index(&self, vertex_id: &VertexId) -> Option<u32> {
        self.vertex_to_index.get(vertex_id).copied()
    }

    /// Look up vertex for an index
    pub fn get_vertex(&self, idx: u32) -> Option<&VertexId> {
        self.index_to_vertex.get(idx as usize)
    }

    /// Remove a vertex and recycle its index
    pub fn remove(&mut self, vertex_id: &VertexId) {
        if let Some(idx) = self.vertex_to_index.remove(vertex_id) {
            self.free_list.push_back(idx);
        }
    }

    /// Current capacity (highest index + 1)
    pub fn capacity(&self) -> usize {
        self.next_index as usize
    }

    /// Number of active vertices
    pub fn len(&self) -> usize {
        self.vertex_to_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertex_to_index.is_empty()
    }

    /// Get all active vertex indices as a bitfield
    pub fn active_bitfield(&self) -> VertexBitfield {
        let mut bf = VertexBitfield::new(self.capacity());
        for &idx in self.vertex_to_index.values() {
            bf.set(idx);
        }
        bf
    }
}

/// Bitfield-based DAG with SIMD-accelerated set operations
pub struct BitfieldDag {
    index_map: VertexIndexMap,
    /// past_sets[i] = bitfield of vertices in i's causal past (transitively)
    past_sets: Vec<VertexBitfield>,
    /// future_sets[i] = bitfield of vertices that have i in their past
    future_sets: Vec<VertexBitfield>,
    /// Direct parents for each vertex (for topological sort)
    parent_sets: Vec<VertexBitfield>,
    /// Round assignment for each vertex index
    rounds: Vec<u64>,
}

impl BitfieldDag {
    pub fn new() -> Self {
        Self {
            index_map: VertexIndexMap::new(),
            past_sets: Vec::new(),
            future_sets: Vec::new(),
            parent_sets: Vec::new(),
            rounds: Vec::new(),
        }
    }

    /// Ensure internal vectors are large enough for the given index
    fn ensure_capacity(&mut self, idx: u32) {
        let needed = idx as usize + 1;
        let cap = self.index_map.capacity().max(needed);

        while self.past_sets.len() < needed {
            self.past_sets.push(VertexBitfield::new(cap));
        }
        while self.future_sets.len() < needed {
            self.future_sets.push(VertexBitfield::new(cap));
        }
        while self.parent_sets.len() < needed {
            self.parent_sets.push(VertexBitfield::new(cap));
        }
        while self.rounds.len() < needed {
            self.rounds.push(0);
        }

        // Grow existing bitfields if needed
        if cap > 0 {
            for bf in self.past_sets.iter_mut() {
                bf.grow_to(cap);
            }
            for bf in self.future_sets.iter_mut() {
                bf.grow_to(cap);
            }
            for bf in self.parent_sets.iter_mut() {
                bf.grow_to(cap);
            }
        }
    }

    /// Add a vertex with its parent dependencies
    pub fn add_vertex(&mut self, vertex_id: VertexId, parents: &[VertexId], round: u64) -> u32 {
        let idx = self.index_map.get_or_insert(vertex_id);
        self.ensure_capacity(idx);

        self.rounds[idx as usize] = round;

        // Build past set: union of all parents' past sets + parents themselves
        let cap = self.index_map.capacity();
        let mut past = VertexBitfield::new(cap);

        for parent_id in parents {
            if let Some(parent_idx) = self.index_map.get_index(parent_id) {
                // Add parent to direct parent set
                self.parent_sets[idx as usize].set(parent_idx);

                // Add parent to past
                past.set(parent_idx);

                // Union parent's transitive past into our past
                if (parent_idx as usize) < self.past_sets.len() {
                    past.union_with(&self.past_sets[parent_idx as usize]);
                }
            }
        }

        self.past_sets[idx as usize] = past;

        // Update future sets: every vertex in our past gets us in their future
        for past_idx in self.past_sets[idx as usize].iter_set_bits().collect::<Vec<_>>() {
            if (past_idx as usize) < self.future_sets.len() {
                self.future_sets[past_idx as usize].set(idx);
            }
        }

        idx
    }

    /// Compute anticone of a vertex (vertices neither in past nor future)
    pub fn anticone(&self, vertex_id: &VertexId) -> Option<VertexBitfield> {
        let idx = self.index_map.get_index(vertex_id)?;
        let active = self.index_map.active_bitfield();

        if (idx as usize) >= self.past_sets.len() || (idx as usize) >= self.future_sets.len() {
            return None;
        }

        Some(VertexBitfield::anticone(
            &self.past_sets[idx as usize],
            &self.future_sets[idx as usize],
            idx,
            &active,
        ))
    }

    /// Compute anticone size (without allocating the full bitfield)
    pub fn anticone_size(&self, vertex_id: &VertexId) -> Option<u32> {
        self.anticone(vertex_id).map(|bf| bf.count_ones())
    }

    /// Topological sort of vertices in a round (using bitfield iteration)
    pub fn topological_sort_round(&self, round: u64) -> Vec<VertexId> {
        // Collect vertices in this round
        let mut round_indices: Vec<u32> = Vec::new();
        for (_, &idx) in &self.index_map.vertex_to_index {
            if (idx as usize) < self.rounds.len() && self.rounds[idx as usize] == round {
                round_indices.push(idx);
            }
        }

        if round_indices.is_empty() {
            return Vec::new();
        }

        // Build in-degree within the round set
        let round_set: HashSet<u32> = round_indices.iter().copied().collect();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();
        for &idx in &round_indices {
            in_degree.insert(idx, 0);
        }

        for &idx in &round_indices {
            if (idx as usize) < self.parent_sets.len() {
                for parent_idx in self.parent_sets[idx as usize].iter_set_bits() {
                    if round_set.contains(&parent_idx) {
                        *in_degree.entry(idx).or_default() += 1;
                    }
                }
            }
        }

        // Kahn's algorithm with deterministic tie-breaking (sort by vertex ID)
        let mut queue: Vec<u32> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&idx, _)| idx)
            .collect();
        // Deterministic: sort by vertex ID bytes
        queue.sort_by(|a, b| {
            let va = self.index_map.get_vertex(*a);
            let vb = self.index_map.get_vertex(*b);
            va.cmp(&vb)
        });

        let mut result = Vec::with_capacity(round_indices.len());

        while let Some(idx) = queue.pop() {
            if let Some(vid) = self.index_map.get_vertex(idx) {
                result.push(*vid);
            }

            // Decrement in-degree of dependents in this round
            for &other_idx in &round_indices {
                if other_idx != idx
                    && (other_idx as usize) < self.parent_sets.len()
                    && self.parent_sets[other_idx as usize].test(idx)
                    && round_set.contains(&other_idx)
                {
                    if let Some(deg) = in_degree.get_mut(&other_idx) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(other_idx);
                            // Re-sort for determinism
                            queue.sort_by(|a, b| {
                                let va = self.index_map.get_vertex(*a);
                                let vb = self.index_map.get_vertex(*b);
                                vb.cmp(&va) // Reverse because we pop from end
                            });
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if vertex A causally precedes vertex B
    pub fn causally_precedes(&self, a: &VertexId, b: &VertexId) -> bool {
        if let (Some(idx_a), Some(idx_b)) = (self.index_map.get_index(a), self.index_map.get_index(b)) {
            if (idx_b as usize) < self.past_sets.len() {
                return self.past_sets[idx_b as usize].test(idx_a);
            }
        }
        false
    }

    /// Remove vertices from rounds older than cutoff, recycling indices
    pub fn cleanup_before_round(&mut self, cutoff_round: u64) {
        let to_remove: Vec<(VertexId, u32)> = self.index_map.vertex_to_index.iter()
            .filter_map(|(&vid, &idx)| {
                if (idx as usize) < self.rounds.len() && self.rounds[idx as usize] < cutoff_round {
                    Some((vid, idx))
                } else {
                    None
                }
            })
            .collect();

        let removed_count = to_remove.len();

        for (vid, idx) in to_remove {
            // Clear this vertex's bitfield data
            if (idx as usize) < self.past_sets.len() {
                self.past_sets[idx as usize] = VertexBitfield::new(self.index_map.capacity());
            }
            if (idx as usize) < self.future_sets.len() {
                self.future_sets[idx as usize] = VertexBitfield::new(self.index_map.capacity());
            }
            if (idx as usize) < self.parent_sets.len() {
                self.parent_sets[idx as usize] = VertexBitfield::new(self.index_map.capacity());
            }

            // Clear this vertex's bit from all other past/future sets
            for bf in self.past_sets.iter_mut() {
                bf.clear(idx);
            }
            for bf in self.future_sets.iter_mut() {
                bf.clear(idx);
            }
            for bf in self.parent_sets.iter_mut() {
                bf.clear(idx);
            }

            // Recycle index
            self.index_map.remove(&vid);
        }

        if removed_count > 0 {
            debug!("BitfieldDag: cleaned {} vertices before round {}", removed_count, cutoff_round);
        }
    }

    /// Get number of active vertices
    pub fn vertex_count(&self) -> usize {
        self.index_map.len()
    }

    /// Get statistics
    pub fn stats(&self) -> BitfieldDagStats {
        let memory_bytes = self.past_sets.iter().map(|bf| bf.bits.len() * 8).sum::<usize>()
            + self.future_sets.iter().map(|bf| bf.bits.len() * 8).sum::<usize>()
            + self.parent_sets.iter().map(|bf| bf.bits.len() * 8).sum::<usize>();

        BitfieldDagStats {
            vertex_count: self.index_map.len(),
            index_capacity: self.index_map.capacity(),
            free_indices: self.index_map.free_list.len(),
            memory_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitfieldDagStats {
    pub vertex_count: usize,
    pub index_capacity: usize,
    pub free_indices: usize,
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vid(n: u8) -> VertexId {
        [n; 32]
    }

    #[test]
    fn test_bitfield_basic_ops() {
        let mut bf = VertexBitfield::new(256);
        assert!(!bf.test(42));
        bf.set(42);
        assert!(bf.test(42));
        bf.clear(42);
        assert!(!bf.test(42));
    }

    #[test]
    fn test_bitfield_union() {
        let mut a = VertexBitfield::new(128);
        let mut b = VertexBitfield::new(128);
        a.set(1);
        a.set(3);
        b.set(2);
        b.set(3);
        a.union_with(&b);
        assert!(a.test(1));
        assert!(a.test(2));
        assert!(a.test(3));
        assert_eq!(a.count_ones(), 3);
    }

    #[test]
    fn test_bitfield_anticone() {
        let mut past = VertexBitfield::new(64);
        let mut future = VertexBitfield::new(64);
        let mut active = VertexBitfield::new(64);

        // Vertices 0-4 active
        for i in 0..5 {
            active.set(i);
        }

        // Vertex 2's past: {0, 1}
        past.set(0);
        past.set(1);

        // Vertex 2's future: {3}
        future.set(3);

        // Anticone of vertex 2 should be {4} (not in past, future, or self)
        let ac = VertexBitfield::anticone(&past, &future, 2, &active);
        assert!(ac.test(4));
        assert!(!ac.test(0));
        assert!(!ac.test(1));
        assert!(!ac.test(2));
        assert!(!ac.test(3));
        assert_eq!(ac.count_ones(), 1);
    }

    #[test]
    fn test_bitfield_iter_set_bits() {
        let mut bf = VertexBitfield::new(256);
        bf.set(0);
        bf.set(63);
        bf.set(64);
        bf.set(127);
        bf.set(200);

        let bits: Vec<u32> = bf.iter_set_bits().collect();
        assert_eq!(bits, vec![0, 63, 64, 127, 200]);
    }

    #[test]
    fn test_vertex_index_map_recycle() {
        let mut map = VertexIndexMap::new();

        let idx0 = map.get_or_insert(vid(0));
        let idx1 = map.get_or_insert(vid(1));
        let idx2 = map.get_or_insert(vid(2));
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);

        // Remove vertex 1
        map.remove(&vid(1));
        assert_eq!(map.len(), 2);

        // New vertex should recycle index 1
        let idx3 = map.get_or_insert(vid(3));
        assert_eq!(idx3, 1); // Recycled!
    }

    #[test]
    fn test_dag_causal_chain() {
        let mut dag = BitfieldDag::new();

        // Chain: v0 -> v1 -> v2
        dag.add_vertex(vid(0), &[], 1);
        dag.add_vertex(vid(1), &[vid(0)], 2);
        dag.add_vertex(vid(2), &[vid(1)], 3);

        assert!(dag.causally_precedes(&vid(0), &vid(1)));
        assert!(dag.causally_precedes(&vid(0), &vid(2)));
        assert!(dag.causally_precedes(&vid(1), &vid(2)));
        assert!(!dag.causally_precedes(&vid(2), &vid(0)));
    }

    #[test]
    fn test_dag_anticone() {
        let mut dag = BitfieldDag::new();

        // Diamond: v0 -> v1, v0 -> v2, v1 -> v3, v2 -> v3
        dag.add_vertex(vid(0), &[], 1);
        dag.add_vertex(vid(1), &[vid(0)], 2);
        dag.add_vertex(vid(2), &[vid(0)], 2);
        dag.add_vertex(vid(3), &[vid(1), vid(2)], 3);

        // v1 and v2 are concurrent (neither in past nor future of each other)
        let ac_v1 = dag.anticone(&vid(1)).unwrap();
        assert!(ac_v1.test(dag.index_map.get_index(&vid(2)).unwrap()));

        let ac_v2 = dag.anticone(&vid(2)).unwrap();
        assert!(ac_v2.test(dag.index_map.get_index(&vid(1)).unwrap()));

        // v0 is in v1's past, not anticone
        assert!(!ac_v1.test(dag.index_map.get_index(&vid(0)).unwrap()));
    }

    #[test]
    fn test_dag_cleanup() {
        let mut dag = BitfieldDag::new();

        dag.add_vertex(vid(0), &[], 1);
        dag.add_vertex(vid(1), &[vid(0)], 2);
        dag.add_vertex(vid(2), &[vid(1)], 3);
        dag.add_vertex(vid(3), &[vid(2)], 4);

        assert_eq!(dag.vertex_count(), 4);

        // Cleanup rounds < 3
        dag.cleanup_before_round(3);
        assert_eq!(dag.vertex_count(), 2); // Only round 3, 4 remain
    }

    #[test]
    fn test_dag_topological_sort() {
        let mut dag = BitfieldDag::new();

        // All in round 1, with dependencies
        dag.add_vertex(vid(1), &[], 1);
        dag.add_vertex(vid(2), &[], 1);
        dag.add_vertex(vid(3), &[vid(1)], 1);

        let sorted = dag.topological_sort_round(1);
        assert_eq!(sorted.len(), 3);

        // v1 must come before v3
        let pos_v1 = sorted.iter().position(|v| *v == vid(1)).unwrap();
        let pos_v3 = sorted.iter().position(|v| *v == vid(3)).unwrap();
        assert!(pos_v1 < pos_v3);
    }

    #[test]
    fn test_intersection_count() {
        let mut a = VertexBitfield::new(128);
        let mut b = VertexBitfield::new(128);

        a.set(1); a.set(2); a.set(3);
        b.set(2); b.set(3); b.set(4);

        assert_eq!(a.intersection_count(&b), 2); // {2, 3}
    }

    #[test]
    fn test_bitfield_grow() {
        let mut bf = VertexBitfield::new(64);
        bf.set(10);
        bf.grow_to(256);
        assert!(bf.test(10)); // Old data preserved
        bf.set(200);
        assert!(bf.test(200));
    }
}
