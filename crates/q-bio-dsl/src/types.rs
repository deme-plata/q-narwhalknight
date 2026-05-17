//! Core types for biological programming
//!
//! Defines fundamental structures for molecules, atoms, bonds,
//! genetic circuits, and synthesis operations.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chemical element with atomic properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Element {
    // Common organic elements
    Hydrogen,
    Carbon,
    Nitrogen,
    Oxygen,
    Phosphorus,
    Sulfur,

    // Halogens
    Fluorine,
    Chlorine,
    Bromine,
    Iodine,

    // Metals (for MOF/ZIF construction)
    Zinc,
    Copper,
    Iron,
    Cobalt,
    Nickel,
    Zirconium,
    Chromium,

    // Other
    Boron,
    Silicon,
    Selenium,
}

impl Element {
    /// Atomic number
    pub fn atomic_number(&self) -> u8 {
        match self {
            Element::Hydrogen => 1,
            Element::Carbon => 6,
            Element::Nitrogen => 7,
            Element::Oxygen => 8,
            Element::Fluorine => 9,
            Element::Phosphorus => 15,
            Element::Sulfur => 16,
            Element::Chlorine => 17,
            Element::Iron => 26,
            Element::Cobalt => 27,
            Element::Nickel => 28,
            Element::Copper => 29,
            Element::Zinc => 30,
            Element::Bromine => 35,
            Element::Zirconium => 40,
            Element::Iodine => 53,
            Element::Boron => 5,
            Element::Silicon => 14,
            Element::Selenium => 34,
            Element::Chromium => 24,
        }
    }

    /// Atomic mass in daltons
    pub fn atomic_mass(&self) -> f64 {
        match self {
            Element::Hydrogen => 1.008,
            Element::Carbon => 12.011,
            Element::Nitrogen => 14.007,
            Element::Oxygen => 15.999,
            Element::Fluorine => 18.998,
            Element::Phosphorus => 30.974,
            Element::Sulfur => 32.065,
            Element::Chlorine => 35.453,
            Element::Iron => 55.845,
            Element::Cobalt => 58.933,
            Element::Nickel => 58.693,
            Element::Copper => 63.546,
            Element::Zinc => 65.38,
            Element::Bromine => 79.904,
            Element::Zirconium => 91.224,
            Element::Iodine => 126.904,
            Element::Boron => 10.811,
            Element::Silicon => 28.086,
            Element::Selenium => 78.971,
            Element::Chromium => 51.996,
        }
    }

    /// Standard valence electrons
    pub fn valence(&self) -> u8 {
        match self {
            Element::Hydrogen => 1,
            Element::Carbon => 4,
            Element::Nitrogen => 3,
            Element::Oxygen => 2,
            Element::Fluorine => 1,
            Element::Phosphorus => 5,
            Element::Sulfur => 6,
            Element::Chlorine => 1,
            Element::Bromine => 1,
            Element::Iodine => 1,
            Element::Boron => 3,
            Element::Silicon => 4,
            Element::Selenium => 6,
            // Metals have variable valence
            _ => 2,
        }
    }

    /// Parse from symbol string
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol.to_uppercase().as_str() {
            "H" => Some(Element::Hydrogen),
            "C" => Some(Element::Carbon),
            "N" => Some(Element::Nitrogen),
            "O" => Some(Element::Oxygen),
            "F" => Some(Element::Fluorine),
            "P" => Some(Element::Phosphorus),
            "S" => Some(Element::Sulfur),
            "CL" => Some(Element::Chlorine),
            "BR" => Some(Element::Bromine),
            "I" => Some(Element::Iodine),
            "ZN" => Some(Element::Zinc),
            "CU" => Some(Element::Copper),
            "FE" => Some(Element::Iron),
            "CO" => Some(Element::Cobalt),
            "NI" => Some(Element::Nickel),
            "ZR" => Some(Element::Zirconium),
            "CR" => Some(Element::Chromium),
            "B" => Some(Element::Boron),
            "SI" => Some(Element::Silicon),
            "SE" => Some(Element::Selenium),
            _ => None,
        }
    }

    /// Symbol string
    pub fn symbol(&self) -> &'static str {
        match self {
            Element::Hydrogen => "H",
            Element::Carbon => "C",
            Element::Nitrogen => "N",
            Element::Oxygen => "O",
            Element::Fluorine => "F",
            Element::Phosphorus => "P",
            Element::Sulfur => "S",
            Element::Chlorine => "Cl",
            Element::Bromine => "Br",
            Element::Iodine => "I",
            Element::Zinc => "Zn",
            Element::Copper => "Cu",
            Element::Iron => "Fe",
            Element::Cobalt => "Co",
            Element::Nickel => "Ni",
            Element::Zirconium => "Zr",
            Element::Chromium => "Cr",
            Element::Boron => "B",
            Element::Silicon => "Si",
            Element::Selenium => "Se",
        }
    }
}

/// Unique identifier for an atom within a molecule
pub type AtomId = u32;

/// Atom with position and properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub id: AtomId,
    pub element: Element,
    pub position: Vector3<f64>,
    pub charge: i8,
    pub isotope: Option<u16>,
    pub hybridization: Hybridization,
}

impl Atom {
    pub fn new(id: AtomId, element: Element, position: Vector3<f64>) -> Self {
        Self {
            id,
            element,
            position,
            charge: 0,
            isotope: None,
            hybridization: Hybridization::default_for(element),
        }
    }
}

/// Orbital hybridization state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Hybridization {
    S,
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
}

impl Hybridization {
    pub fn default_for(element: Element) -> Self {
        match element {
            Element::Carbon => Hybridization::SP3,
            Element::Nitrogen => Hybridization::SP3,
            Element::Oxygen => Hybridization::SP3,
            Element::Phosphorus => Hybridization::SP3,
            Element::Sulfur => Hybridization::SP3,
            _ => Hybridization::S,
        }
    }
}

/// Chemical bond type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Ionic,
    Hydrogen,
    Coordinate, // Dative bond
}

impl BondType {
    pub fn order(&self) -> f64 {
        match self {
            BondType::Single => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
            BondType::Ionic => 0.5,
            BondType::Hydrogen => 0.1,
            BondType::Coordinate => 1.0,
        }
    }

    pub fn bond_length(&self, elem1: Element, elem2: Element) -> f64 {
        // Approximate bond lengths in Angstroms
        let base = match (elem1, elem2) {
            (Element::Carbon, Element::Carbon) => 1.54,
            (Element::Carbon, Element::Hydrogen) => 1.09,
            (Element::Carbon, Element::Oxygen) => 1.43,
            (Element::Carbon, Element::Nitrogen) => 1.47,
            (Element::Oxygen, Element::Hydrogen) => 0.96,
            (Element::Nitrogen, Element::Hydrogen) => 1.01,
            _ => 1.5,
        };

        match self {
            BondType::Single => base,
            BondType::Double => base * 0.87,
            BondType::Triple => base * 0.78,
            BondType::Aromatic => base * 0.91,
            _ => base,
        }
    }
}

/// Chemical bond between two atoms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bond {
    pub atom1: AtomId,
    pub atom2: AtomId,
    pub bond_type: BondType,
    pub stereo: Option<BondStereo>,
}

/// Bond stereochemistry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondStereo {
    None,
    Up,   // Wedge
    Down, // Dash
    CisOrTrans,
    Either,
}

/// Stereocenter configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StereoConfig {
    R, // Rectus
    S, // Sinister
    E, // Entgegen (trans)
    Z, // Zusammen (cis)
}

/// Stereocenter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stereocenter {
    pub center_atom: AtomId,
    pub config: StereoConfig,
    pub neighbors: Vec<AtomId>,
}

/// Complete molecular structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Molecule {
    pub name: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub stereocenters: Vec<Stereocenter>,
    pub smiles: Option<String>,
    pub molecular_formula: Option<String>,
    pub molecular_weight: Option<f64>,
    pub properties: MoleculeProperties,
}

impl Molecule {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            stereocenters: Vec::new(),
            smiles: None,
            molecular_formula: None,
            molecular_weight: None,
            properties: MoleculeProperties::default(),
        }
    }

    /// Add an atom and return its ID
    pub fn add_atom(&mut self, element: Element, position: Vector3<f64>) -> AtomId {
        let id = self.atoms.len() as AtomId;
        self.atoms.push(Atom::new(id, element, position));
        id
    }

    /// Add a bond between two atoms
    pub fn add_bond(&mut self, atom1: AtomId, atom2: AtomId, bond_type: BondType) {
        self.bonds.push(Bond {
            atom1,
            atom2,
            bond_type,
            stereo: None,
        });
    }

    /// Calculate molecular formula
    pub fn calculate_formula(&mut self) {
        let mut counts: HashMap<Element, u32> = HashMap::new();
        for atom in &self.atoms {
            *counts.entry(atom.element).or_insert(0) += 1;
        }

        // Standard Hill notation (C first, then H, then alphabetical)
        let mut formula = String::new();
        if let Some(&c) = counts.get(&Element::Carbon) {
            formula.push('C');
            if c > 1 {
                formula.push_str(&c.to_string());
            }
            counts.remove(&Element::Carbon);
        }
        if let Some(&h) = counts.get(&Element::Hydrogen) {
            formula.push('H');
            if h > 1 {
                formula.push_str(&h.to_string());
            }
            counts.remove(&Element::Hydrogen);
        }

        let mut remaining: Vec<_> = counts.iter().collect();
        remaining.sort_by_key(|(e, _)| e.symbol());
        for (element, count) in remaining {
            formula.push_str(element.symbol());
            if *count > 1 {
                formula.push_str(&count.to_string());
            }
        }

        self.molecular_formula = Some(formula);
    }

    /// Calculate molecular weight
    pub fn calculate_weight(&mut self) {
        let weight: f64 = self.atoms.iter().map(|a| a.element.atomic_mass()).sum();
        self.molecular_weight = Some(weight);
    }

    /// Get atom by ID
    pub fn get_atom(&self, id: AtomId) -> Option<&Atom> {
        self.atoms.iter().find(|a| a.id == id)
    }

    /// Get bonds for an atom
    pub fn get_bonds_for_atom(&self, atom_id: AtomId) -> Vec<&Bond> {
        self.bonds
            .iter()
            .filter(|b| b.atom1 == atom_id || b.atom2 == atom_id)
            .collect()
    }

    /// Count atoms by element
    pub fn count_element(&self, element: Element) -> usize {
        self.atoms.iter().filter(|a| a.element == element).count()
    }
}

/// Additional molecular properties
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoleculeProperties {
    pub log_p: Option<f64>,           // Lipophilicity
    pub pka: Option<f64>,             // Acid dissociation constant
    pub solubility: Option<f64>,      // Water solubility (mg/mL)
    pub melting_point: Option<f64>,   // Celsius
    pub boiling_point: Option<f64>,   // Celsius
    pub density: Option<f64>,         // g/cm³
    pub dipole_moment: Option<f64>,   // Debye
    pub rotatable_bonds: Option<u32>,
    pub h_bond_donors: Option<u32>,
    pub h_bond_acceptors: Option<u32>,
    pub polar_surface_area: Option<f64>, // Å²
}

/// Ring structure in a molecule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ring {
    pub atoms: Vec<AtomId>,
    pub ring_type: RingType,
    pub aromatic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RingType {
    Cyclopropane,
    Cyclobutane,
    Cyclopentane,
    Cyclohexane,
    Benzene,
    Pyridine,
    Pyrrole,
    Furan,
    Thiophene,
    Imidazole,
    Pyrimidine,
    Purine,
    Indole,
    Custom(usize), // Size
}

/// DNA/RNA nucleotide base
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Nucleotide {
    Adenine,
    Thymine,
    Guanine,
    Cytosine,
    Uracil, // RNA only
}

impl Nucleotide {
    pub fn symbol(&self) -> char {
        match self {
            Nucleotide::Adenine => 'A',
            Nucleotide::Thymine => 'T',
            Nucleotide::Guanine => 'G',
            Nucleotide::Cytosine => 'C',
            Nucleotide::Uracil => 'U',
        }
    }

    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'A' => Some(Nucleotide::Adenine),
            'T' => Some(Nucleotide::Thymine),
            'G' => Some(Nucleotide::Guanine),
            'C' => Some(Nucleotide::Cytosine),
            'U' => Some(Nucleotide::Uracil),
            _ => None,
        }
    }

    pub fn complement(&self) -> Self {
        match self {
            Nucleotide::Adenine => Nucleotide::Thymine,
            Nucleotide::Thymine => Nucleotide::Adenine,
            Nucleotide::Guanine => Nucleotide::Cytosine,
            Nucleotide::Cytosine => Nucleotide::Guanine,
            Nucleotide::Uracil => Nucleotide::Adenine,
        }
    }
}

/// DNA sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNASequence {
    pub sequence: Vec<Nucleotide>,
    pub name: Option<String>,
    pub modifications: Vec<DNAModification>,
}

impl DNASequence {
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
            name: None,
            modifications: Vec::new(),
        }
    }

    pub fn from_string(s: &str) -> Option<Self> {
        let sequence: Option<Vec<Nucleotide>> = s.chars().map(Nucleotide::from_char).collect();
        Some(Self {
            sequence: sequence?,
            name: None,
            modifications: Vec::new(),
        })
    }

    pub fn to_string(&self) -> String {
        self.sequence.iter().map(|n| n.symbol()).collect()
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    pub fn append(&mut self, other: &DNASequence) {
        self.sequence.extend(other.sequence.iter().cloned());
    }

    pub fn complement(&self) -> Self {
        Self {
            sequence: self.sequence.iter().map(|n| n.complement()).collect(),
            name: self.name.clone().map(|n| format!("{}_complement", n)),
            modifications: Vec::new(),
        }
    }

    pub fn reverse_complement(&self) -> Self {
        Self {
            sequence: self
                .sequence
                .iter()
                .rev()
                .map(|n| n.complement())
                .collect(),
            name: self.name.clone().map(|n| format!("{}_revcomp", n)),
            modifications: Vec::new(),
        }
    }
}

impl Default for DNASequence {
    fn default() -> Self {
        Self::new()
    }
}

/// DNA modification (methylation, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DNAModification {
    pub position: usize,
    pub modification_type: DNAModificationType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DNAModificationType {
    Methylation5C,      // 5-methylcytosine
    Methylation6A,      // N6-methyladenine
    Phosphorothioate,   // Backbone modification
    LNA,                // Locked nucleic acid
    BNA,                // Bridged nucleic acid
    Fluorescent(String), // Fluorescent label
}

/// Amino acid for protein sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    Alanine,       // A
    Arginine,      // R
    Asparagine,    // N
    AsparticAcid,  // D
    Cysteine,      // C
    GlutamicAcid,  // E
    Glutamine,     // Q
    Glycine,       // G
    Histidine,     // H
    Isoleucine,    // I
    Leucine,       // L
    Lysine,        // K
    Methionine,    // M
    Phenylalanine, // F
    Proline,       // P
    Serine,        // S
    Threonine,     // T
    Tryptophan,    // W
    Tyrosine,      // Y
    Valine,        // V
    Stop,          // *
}

impl AminoAcid {
    pub fn one_letter(&self) -> char {
        match self {
            AminoAcid::Alanine => 'A',
            AminoAcid::Arginine => 'R',
            AminoAcid::Asparagine => 'N',
            AminoAcid::AsparticAcid => 'D',
            AminoAcid::Cysteine => 'C',
            AminoAcid::GlutamicAcid => 'E',
            AminoAcid::Glutamine => 'Q',
            AminoAcid::Glycine => 'G',
            AminoAcid::Histidine => 'H',
            AminoAcid::Isoleucine => 'I',
            AminoAcid::Leucine => 'L',
            AminoAcid::Lysine => 'K',
            AminoAcid::Methionine => 'M',
            AminoAcid::Phenylalanine => 'F',
            AminoAcid::Proline => 'P',
            AminoAcid::Serine => 'S',
            AminoAcid::Threonine => 'T',
            AminoAcid::Tryptophan => 'W',
            AminoAcid::Tyrosine => 'Y',
            AminoAcid::Valine => 'V',
            AminoAcid::Stop => '*',
        }
    }

    pub fn three_letter(&self) -> &'static str {
        match self {
            AminoAcid::Alanine => "Ala",
            AminoAcid::Arginine => "Arg",
            AminoAcid::Asparagine => "Asn",
            AminoAcid::AsparticAcid => "Asp",
            AminoAcid::Cysteine => "Cys",
            AminoAcid::GlutamicAcid => "Glu",
            AminoAcid::Glutamine => "Gln",
            AminoAcid::Glycine => "Gly",
            AminoAcid::Histidine => "His",
            AminoAcid::Isoleucine => "Ile",
            AminoAcid::Leucine => "Leu",
            AminoAcid::Lysine => "Lys",
            AminoAcid::Methionine => "Met",
            AminoAcid::Phenylalanine => "Phe",
            AminoAcid::Proline => "Pro",
            AminoAcid::Serine => "Ser",
            AminoAcid::Threonine => "Thr",
            AminoAcid::Tryptophan => "Trp",
            AminoAcid::Tyrosine => "Tyr",
            AminoAcid::Valine => "Val",
            AminoAcid::Stop => "Stop",
        }
    }

    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'A' => Some(AminoAcid::Alanine),
            'R' => Some(AminoAcid::Arginine),
            'N' => Some(AminoAcid::Asparagine),
            'D' => Some(AminoAcid::AsparticAcid),
            'C' => Some(AminoAcid::Cysteine),
            'E' => Some(AminoAcid::GlutamicAcid),
            'Q' => Some(AminoAcid::Glutamine),
            'G' => Some(AminoAcid::Glycine),
            'H' => Some(AminoAcid::Histidine),
            'I' => Some(AminoAcid::Isoleucine),
            'L' => Some(AminoAcid::Leucine),
            'K' => Some(AminoAcid::Lysine),
            'M' => Some(AminoAcid::Methionine),
            'F' => Some(AminoAcid::Phenylalanine),
            'P' => Some(AminoAcid::Proline),
            'S' => Some(AminoAcid::Serine),
            'T' => Some(AminoAcid::Threonine),
            'W' => Some(AminoAcid::Tryptophan),
            'Y' => Some(AminoAcid::Tyrosine),
            'V' => Some(AminoAcid::Valine),
            '*' => Some(AminoAcid::Stop),
            _ => None,
        }
    }

    pub fn molecular_weight(&self) -> f64 {
        match self {
            AminoAcid::Alanine => 89.09,
            AminoAcid::Arginine => 174.20,
            AminoAcid::Asparagine => 132.12,
            AminoAcid::AsparticAcid => 133.10,
            AminoAcid::Cysteine => 121.15,
            AminoAcid::GlutamicAcid => 147.13,
            AminoAcid::Glutamine => 146.15,
            AminoAcid::Glycine => 75.07,
            AminoAcid::Histidine => 155.16,
            AminoAcid::Isoleucine => 131.17,
            AminoAcid::Leucine => 131.17,
            AminoAcid::Lysine => 146.19,
            AminoAcid::Methionine => 149.21,
            AminoAcid::Phenylalanine => 165.19,
            AminoAcid::Proline => 115.13,
            AminoAcid::Serine => 105.09,
            AminoAcid::Threonine => 119.12,
            AminoAcid::Tryptophan => 204.23,
            AminoAcid::Tyrosine => 181.19,
            AminoAcid::Valine => 117.15,
            AminoAcid::Stop => 0.0,
        }
    }
}

/// Protein sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinSequence {
    pub sequence: Vec<AminoAcid>,
    pub name: Option<String>,
}

impl ProteinSequence {
    pub fn from_string(s: &str) -> Option<Self> {
        let sequence: Option<Vec<AminoAcid>> = s.chars().map(AminoAcid::from_char).collect();
        Some(Self {
            sequence: sequence?,
            name: None,
        })
    }

    pub fn to_string(&self) -> String {
        self.sequence.iter().map(|a| a.one_letter()).collect()
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    pub fn molecular_weight(&self) -> f64 {
        // Sum weights minus water for each peptide bond
        let sum: f64 = self.sequence.iter().map(|a| a.molecular_weight()).sum();
        sum - (self.sequence.len().saturating_sub(1) as f64 * 18.015)
    }
}

/// Quantity units for synthesis
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantityUnit {
    Nanograms,
    Micrograms,
    Milligrams,
    Grams,
    Moles,
    Millimoles,
    Micromoles,
    Nanomoles,
    Molecules, // Absolute count
}

impl QuantityUnit {
    pub fn to_grams(&self, amount: f64, molecular_weight: f64) -> f64 {
        match self {
            QuantityUnit::Nanograms => amount * 1e-9,
            QuantityUnit::Micrograms => amount * 1e-6,
            QuantityUnit::Milligrams => amount * 1e-3,
            QuantityUnit::Grams => amount,
            QuantityUnit::Moles => amount * molecular_weight,
            QuantityUnit::Millimoles => amount * molecular_weight * 1e-3,
            QuantityUnit::Micromoles => amount * molecular_weight * 1e-6,
            QuantityUnit::Nanomoles => amount * molecular_weight * 1e-9,
            QuantityUnit::Molecules => (amount / 6.022e23) * molecular_weight,
        }
    }
}

/// Synthesis quantity specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantity {
    pub amount: f64,
    pub unit: QuantityUnit,
}

impl Quantity {
    pub fn milligrams(amount: f64) -> Self {
        Self {
            amount,
            unit: QuantityUnit::Milligrams,
        }
    }

    pub fn micrograms(amount: f64) -> Self {
        Self {
            amount,
            unit: QuantityUnit::Micrograms,
        }
    }

    pub fn nanograms(amount: f64) -> Self {
        Self {
            amount,
            unit: QuantityUnit::Nanograms,
        }
    }

    pub fn moles(amount: f64) -> Self {
        Self {
            amount,
            unit: QuantityUnit::Moles,
        }
    }

    pub fn molecules(count: f64) -> Self {
        Self {
            amount: count,
            unit: QuantityUnit::Molecules,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_properties() {
        assert_eq!(Element::Carbon.atomic_number(), 6);
        assert_eq!(Element::Carbon.valence(), 4);
        assert!((Element::Carbon.atomic_mass() - 12.011).abs() < 0.001);
    }

    #[test]
    fn test_dna_sequence() {
        let dna = DNASequence::from_string("ATCG").unwrap();
        assert_eq!(dna.len(), 4);
        assert_eq!(dna.to_string(), "ATCG");

        let complement = dna.complement();
        assert_eq!(complement.to_string(), "TAGC");
    }

    #[test]
    fn test_protein_sequence() {
        let protein = ProteinSequence::from_string("MVLSPADKTNVK").unwrap();
        assert_eq!(protein.len(), 12);
        assert!(protein.molecular_weight() > 0.0);
    }

    #[test]
    fn test_molecule_formula() {
        let mut mol = Molecule::new("Methane");
        let c = mol.add_atom(Element::Carbon, Vector3::zeros());
        for i in 0..4 {
            let h = mol.add_atom(
                Element::Hydrogen,
                Vector3::new(1.09 * (i as f64).cos(), 1.09 * (i as f64).sin(), 0.0),
            );
            mol.add_bond(c, h, BondType::Single);
        }
        mol.calculate_formula();
        assert_eq!(mol.molecular_formula, Some("CH4".to_string()));
    }
}
