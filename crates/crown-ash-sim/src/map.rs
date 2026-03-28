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
}
