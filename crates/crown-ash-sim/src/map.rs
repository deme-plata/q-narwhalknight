//! Fixed adjacency graph for the 25-province campaign map.
//!
//! The map is divided into 7 regional clusters (one per starting faction)
//! with inter-region borders creating strategic chokepoints.
//!
//! Movement cost is `base_cost * terrain.movement_cost()`.

use crown_ash_types::{FixedPoint, Terrain};

/// Base movement cost for a single province-to-province step (1.000 turns).
pub const BASE_MOVEMENT_COST: FixedPoint = FixedPoint::from_int(1);

/// Province adjacency data: `(province_id, name, terrain, faction_owner, [neighbor_ids])`.
///
/// Layout (rough geographical clusters):
///
/// ```text
///   FROST MARCHES (north)                    RED STEPPE (northeast)
///     [0] [1] [2] [3]                          [21] [22] [23] [24]
///          \   |   /                               |    |
///   BLACK ABBEY (west)    ASHEN CROWN (center)
///     [18] [19] [20]      [7] [8] [9] [10]
///          |              /        \
///   EMBER CHURCH (sw)   VALE PRINCES (se)        SALT LEAGUE (east)
///     [11] [12] [13]    [4] [5] [6]              [14] [15] [16] [17]
/// ```
pub struct ProvinceData {
    pub id: u16,
    pub name: &'static str,
    pub terrain: Terrain,
    pub starting_faction: u8,
    pub neighbors: &'static [u16],
}

/// All 25 provinces with their fixed adjacency.
pub const PROVINCE_DATA: [ProvinceData; 25] = [
    // --- Frost Marches (faction 4) --- provinces 0-3
    ProvinceData { id: 0,  name: "Frosthold",        terrain: Terrain::Mountains, starting_faction: 4, neighbors: &[1, 2, 18] },
    ProvinceData { id: 1,  name: "Winterfell Vale",   terrain: Terrain::Hills,     starting_faction: 4, neighbors: &[0, 2, 7] },
    ProvinceData { id: 2,  name: "Icemere",           terrain: Terrain::Marsh,     starting_faction: 4, neighbors: &[0, 1, 3, 8] },
    ProvinceData { id: 3,  name: "Stormwatch",        terrain: Terrain::Coastal,   starting_faction: 4, neighbors: &[2, 21, 8] },
    // --- Vale Princes (faction 1) --- provinces 4-6
    ProvinceData { id: 4,  name: "Goldhaven",         terrain: Terrain::Plains,    starting_faction: 1, neighbors: &[5, 9, 13] },
    ProvinceData { id: 5,  name: "Thornwall",         terrain: Terrain::Hills,     starting_faction: 1, neighbors: &[4, 6, 10, 14] },
    ProvinceData { id: 6,  name: "Ravensgate",        terrain: Terrain::Forest,    starting_faction: 1, neighbors: &[5, 15, 10] },
    // --- Ashen Crown (faction 0) --- provinces 7-10
    ProvinceData { id: 7,  name: "Ashenmere",         terrain: Terrain::Plains,    starting_faction: 0, neighbors: &[1, 8, 11, 9] },
    ProvinceData { id: 8,  name: "Crownspire",        terrain: Terrain::Hills,     starting_faction: 0, neighbors: &[2, 3, 7, 9, 20, 21] },
    ProvinceData { id: 9,  name: "Embervale",         terrain: Terrain::River,     starting_faction: 0, neighbors: &[7, 8, 10, 4] },
    ProvinceData { id: 10, name: "Kingsreach",        terrain: Terrain::Plains,    starting_faction: 0, neighbors: &[9, 5, 6, 12] },
    // --- Ember Church (faction 2) --- provinces 11-13
    ProvinceData { id: 11, name: "Sanctum",           terrain: Terrain::Hills,     starting_faction: 2, neighbors: &[7, 12, 19] },
    ProvinceData { id: 12, name: "Pyrelight",         terrain: Terrain::Plains,    starting_faction: 2, neighbors: &[11, 13, 10] },
    ProvinceData { id: 13, name: "Candlekeep",        terrain: Terrain::Forest,    starting_faction: 2, neighbors: &[12, 4, 17] },
    // --- Salt League (faction 3) --- provinces 14-17
    ProvinceData { id: 14, name: "Saltmere",          terrain: Terrain::Coastal,   starting_faction: 3, neighbors: &[5, 15, 16] },
    ProvinceData { id: 15, name: "Tidehollow",        terrain: Terrain::Coastal,   starting_faction: 3, neighbors: &[6, 14, 16] },
    ProvinceData { id: 16, name: "Coinport",          terrain: Terrain::Coastal,   starting_faction: 3, neighbors: &[14, 15, 17] },
    ProvinceData { id: 17, name: "Warehouse Row",     terrain: Terrain::Plains,    starting_faction: 3, neighbors: &[16, 13, 24] },
    // --- Black Abbey (faction 6) --- provinces 18-20
    ProvinceData { id: 18, name: "Shadowmere",        terrain: Terrain::Forest,    starting_faction: 6, neighbors: &[0, 19, 20] },
    ProvinceData { id: 19, name: "Whispering Cloister", terrain: Terrain::Hills,   starting_faction: 6, neighbors: &[18, 11, 20] },
    ProvinceData { id: 20, name: "Veilstone",         terrain: Terrain::Mountains, starting_faction: 6, neighbors: &[18, 19, 8] },
    // --- Red Steppe (faction 5) --- provinces 21-24
    ProvinceData { id: 21, name: "Khanstead",         terrain: Terrain::Plains,    starting_faction: 5, neighbors: &[3, 22, 8] },
    ProvinceData { id: 22, name: "Windbreak",         terrain: Terrain::Desert,    starting_faction: 5, neighbors: &[21, 23, 24] },
    ProvinceData { id: 23, name: "Dustmane",          terrain: Terrain::Desert,    starting_faction: 5, neighbors: &[22, 24] },
    ProvinceData { id: 24, name: "Redhorn",           terrain: Terrain::Plains,    starting_faction: 5, neighbors: &[22, 23, 17] },
];

/// Get the neighbor list for a given province.
pub fn neighbors(province_id: u16) -> &'static [u16] {
    if (province_id as usize) < PROVINCE_DATA.len() {
        PROVINCE_DATA[province_id as usize].neighbors
    } else {
        &[]
    }
}

/// Check whether two provinces are adjacent.
pub fn are_adjacent(a: u16, b: u16) -> bool {
    neighbors(a).contains(&b)
}

/// Movement cost from province `from` to adjacent province `to`.
///
/// Returns `None` if the provinces are not adjacent.
/// Cost = `BASE_MOVEMENT_COST * destination_terrain.movement_cost()`.
pub fn movement_cost(from: u16, to: u16) -> Option<FixedPoint> {
    if !are_adjacent(from, to) {
        return None;
    }
    if (to as usize) < PROVINCE_DATA.len() {
        let terrain = PROVINCE_DATA[to as usize].terrain;
        Some(BASE_MOVEMENT_COST.mul_fp(terrain.movement_cost()))
    } else {
        None
    }
}

/// BFS shortest path from `start` to `goal`.
///
/// Returns the path as a sequence of province IDs **excluding** `start`,
/// or `None` if no path exists.  The first element of the returned Vec
/// is the immediate next step.
///
/// Complexity: O(V + E) where V=25, E≈63 — trivially cheap.
pub fn find_path(start: u16, goal: u16) -> Option<Vec<u16>> {
    if start == goal {
        return Some(Vec::new());
    }
    if start as usize >= PROVINCE_DATA.len() || goal as usize >= PROVINCE_DATA.len() {
        return None;
    }

    // BFS with parent tracking.
    let mut visited = [false; 25];
    let mut parent = [u16::MAX; 25];
    let mut queue = std::collections::VecDeque::new();

    visited[start as usize] = true;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        for &neighbor in neighbors(current) {
            let ni = neighbor as usize;
            if ni < 25 && !visited[ni] {
                visited[ni] = true;
                parent[ni] = current;
                if neighbor == goal {
                    // Reconstruct path.
                    let mut path = Vec::new();
                    let mut node = goal;
                    while node != start {
                        path.push(node);
                        node = parent[node as usize];
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(neighbor);
            }
        }
    }

    None // disconnected (shouldn't happen with our map)
}

/// BFS path length (number of hops) from `start` to `goal`.
pub fn path_length(start: u16, goal: u16) -> Option<usize> {
    find_path(start, goal).map(|p| p.len())
}

/// Get terrain for a province.
pub fn terrain(province_id: u16) -> Terrain {
    if (province_id as usize) < PROVINCE_DATA.len() {
        PROVINCE_DATA[province_id as usize].terrain
    } else {
        Terrain::Plains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_is_symmetric() {
        for data in &PROVINCE_DATA {
            for &neighbor in data.neighbors {
                assert!(
                    are_adjacent(neighbor, data.id),
                    "Province {} lists {} as neighbor but not vice versa",
                    data.id, neighbor
                );
            }
        }
    }

    #[test]
    fn all_provinces_have_neighbors() {
        for data in &PROVINCE_DATA {
            assert!(!data.neighbors.is_empty(), "Province {} has no neighbors", data.id);
        }
    }

    #[test]
    fn no_self_adjacency() {
        for data in &PROVINCE_DATA {
            assert!(
                !data.neighbors.contains(&data.id),
                "Province {} is adjacent to itself",
                data.id
            );
        }
    }

    #[test]
    fn movement_cost_plains() {
        // Plains to Plains should be BASE_MOVEMENT_COST (1.000)
        // Province 7 (Plains) → Province 9 (River, cost 1.200)
        let cost = movement_cost(7, 9).unwrap();
        assert_eq!(cost.raw(), 1200);
    }

    #[test]
    fn find_path_adjacent() {
        // Direct neighbors should return a 1-element path.
        let path = find_path(7, 8).unwrap();
        assert_eq!(path, vec![8]);
    }

    #[test]
    fn find_path_same_province() {
        let path = find_path(5, 5).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn find_path_multi_hop() {
        // Province 0 (Frosthold) → Province 9 (Embervale)
        // Must go through at least 1→7→9 or similar.
        let path = find_path(0, 9).unwrap();
        assert!(path.len() >= 2, "Path should be at least 2 hops");
        // Verify each step is adjacent to the previous.
        let mut prev = 0u16;
        for &step in &path {
            assert!(are_adjacent(prev, step), "{} not adjacent to {}", prev, step);
            prev = step;
        }
        assert_eq!(*path.last().unwrap(), 9);
    }

    #[test]
    fn find_path_cross_map() {
        // Province 18 (Shadowmere, far west) → Province 24 (Redhorn, far east)
        let path = find_path(18, 24).unwrap();
        assert!(path.len() >= 3, "Cross-map path should be 3+ hops");
        let mut prev = 18u16;
        for &step in &path {
            assert!(are_adjacent(prev, step));
            prev = step;
        }
        assert_eq!(*path.last().unwrap(), 24);
    }

    #[test]
    fn path_length_works() {
        assert_eq!(path_length(7, 7), Some(0));
        assert_eq!(path_length(7, 8), Some(1));
        assert!(path_length(0, 24).unwrap() >= 3);
    }
}
