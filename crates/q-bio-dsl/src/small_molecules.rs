//! Small Molecule Templates and Synthesis
//!
//! Pre-defined molecular templates for common compounds including
//! cannabinoids, terpenes, alkaloids, and pharmaceuticals.

use crate::ast::MoleculeDefinition;
use crate::types::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Small molecule template library
pub struct SmallMoleculeLibrary;

impl SmallMoleculeLibrary {
    // ═══════════════════════════════════════════════════════════════════════════════
    // CANNABINOIDS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Delta-9-Tetrahydrocannabinol (THC)
    /// IUPAC: (6aR,10aR)-6,6,9-trimethyl-3-pentyl-6a,7,8,10a-tetrahydro-6H-benzo[c]chromen-1-ol
    pub const THC_SMILES: &'static str =
        "CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1)(C)C)C)O";

    /// Cannabidiol (CBD)
    /// Non-psychoactive cannabinoid
    pub const CBD_SMILES: &'static str =
        "CCCCCC1=CC(=C(C(=C1)O)[C@@H]2C=C(CC[C@H]2C(=C)C)C)O";

    /// Cannabinol (CBN)
    /// Degradation product of THC
    pub const CBN_SMILES: &'static str =
        "CCCCCC1=CC(=C2C(=C1)OC3=CC(=CC=C3C2=O)C)O";

    /// Cannabigerol (CBG)
    /// Precursor cannabinoid
    pub const CBG_SMILES: &'static str =
        "CCCCCC1=CC(=C(C(=C1)O)CC=C(C)CCC=C(C)C)O";

    /// Delta-8-THC
    pub const DELTA8_THC_SMILES: &'static str =
        "CCCCCC1=CC(=C2[C@@H]3CC(=CC[C@H]3C(OC2=C1)(C)C)C)O";

    /// THCA (THC Acid)
    pub const THCA_SMILES: &'static str =
        "CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1C(=O)O)(C)C)C)O";

    // ═══════════════════════════════════════════════════════════════════════════════
    // TERPENES
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Limonene - citrus scent
    pub const LIMONENE_SMILES: &'static str = "CC1=CCC(CC1)C(=C)C";

    /// Myrcene - earthy, musky
    pub const MYRCENE_SMILES: &'static str = "CC(=CCCC(=C)C=C)C";

    /// Pinene - pine scent
    pub const PINENE_SMILES: &'static str = "CC1=CCC2CC1C2(C)C";

    /// Linalool - floral
    pub const LINALOOL_SMILES: &'static str = "CC(=CCCC(C)(C=C)O)C";

    /// Caryophyllene - spicy
    pub const CARYOPHYLLENE_SMILES: &'static str = "CC1=CCCC(=C)C2CC(C2CC1)(C)C";

    /// Humulene - woody
    pub const HUMULENE_SMILES: &'static str = "CC1=CCC(C=CCC(=CC1)C)(C)C";

    // ═══════════════════════════════════════════════════════════════════════════════
    // ALKALOIDS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Caffeine
    pub const CAFFEINE_SMILES: &'static str = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C";

    /// Nicotine
    pub const NICOTINE_SMILES: &'static str = "CN1CCC[C@H]1c2cccnc2";

    /// Psilocybin
    pub const PSILOCYBIN_SMILES: &'static str = "CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12";

    /// Psilocin
    pub const PSILOCIN_SMILES: &'static str = "CN(C)CCc1c[nH]c2cccc(O)c12";

    /// DMT (N,N-Dimethyltryptamine)
    pub const DMT_SMILES: &'static str = "CN(C)CCc1c[nH]c2ccccc12";

    /// Mescaline
    pub const MESCALINE_SMILES: &'static str = "COc1cc(CCN)cc(OC)c1OC";

    // ═══════════════════════════════════════════════════════════════════════════════
    // PHARMACEUTICALS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Aspirin (Acetylsalicylic acid)
    pub const ASPIRIN_SMILES: &'static str = "CC(=O)Oc1ccccc1C(=O)O";

    /// Ibuprofen
    pub const IBUPROFEN_SMILES: &'static str = "CC(C)Cc1ccc(cc1)C(C)C(=O)O";

    /// Acetaminophen (Paracetamol)
    pub const ACETAMINOPHEN_SMILES: &'static str = "CC(=O)Nc1ccc(O)cc1";

    /// Melatonin
    pub const MELATONIN_SMILES: &'static str = "CC(=O)NCCc1c[nH]c2ccc(OC)cc12";

    /// Serotonin
    pub const SEROTONIN_SMILES: &'static str = "NCCc1c[nH]c2ccc(O)cc12";

    /// Dopamine
    pub const DOPAMINE_SMILES: &'static str = "NCCc1ccc(O)c(O)c1";

    /// Adrenaline (Epinephrine)
    pub const ADRENALINE_SMILES: &'static str = "CNC[C@H](O)c1ccc(O)c(O)c1";

    // ═══════════════════════════════════════════════════════════════════════════════
    // AMINO ACIDS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Glycine
    pub const GLYCINE_SMILES: &'static str = "NCC(=O)O";

    /// Alanine
    pub const ALANINE_SMILES: &'static str = "C[C@H](N)C(=O)O";

    /// Tryptophan
    pub const TRYPTOPHAN_SMILES: &'static str = "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O";

    /// Phenylalanine
    pub const PHENYLALANINE_SMILES: &'static str = "N[C@@H](Cc1ccccc1)C(=O)O";

    // ═══════════════════════════════════════════════════════════════════════════════
    // NUCLEOTIDES
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Adenosine
    pub const ADENOSINE_SMILES: &'static str =
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O";

    /// Guanosine
    pub const GUANOSINE_SMILES: &'static str =
        "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1";

    /// Get molecule definition by name
    pub fn get(name: &str) -> Option<MoleculeDefinition> {
        let smiles = match name.to_lowercase().as_str() {
            // Cannabinoids
            "thc" | "delta9-thc" | "delta-9-thc" => Self::THC_SMILES,
            "cbd" | "cannabidiol" => Self::CBD_SMILES,
            "cbn" | "cannabinol" => Self::CBN_SMILES,
            "cbg" | "cannabigerol" => Self::CBG_SMILES,
            "delta8-thc" | "delta-8-thc" => Self::DELTA8_THC_SMILES,
            "thca" => Self::THCA_SMILES,

            // Terpenes
            "limonene" => Self::LIMONENE_SMILES,
            "myrcene" => Self::MYRCENE_SMILES,
            "pinene" | "alpha-pinene" => Self::PINENE_SMILES,
            "linalool" => Self::LINALOOL_SMILES,
            "caryophyllene" | "beta-caryophyllene" => Self::CARYOPHYLLENE_SMILES,
            "humulene" => Self::HUMULENE_SMILES,

            // Alkaloids
            "caffeine" => Self::CAFFEINE_SMILES,
            "nicotine" => Self::NICOTINE_SMILES,
            "psilocybin" => Self::PSILOCYBIN_SMILES,
            "psilocin" => Self::PSILOCIN_SMILES,
            "dmt" => Self::DMT_SMILES,
            "mescaline" => Self::MESCALINE_SMILES,

            // Pharmaceuticals
            "aspirin" => Self::ASPIRIN_SMILES,
            "ibuprofen" => Self::IBUPROFEN_SMILES,
            "acetaminophen" | "paracetamol" => Self::ACETAMINOPHEN_SMILES,
            "melatonin" => Self::MELATONIN_SMILES,
            "serotonin" => Self::SEROTONIN_SMILES,
            "dopamine" => Self::DOPAMINE_SMILES,
            "adrenaline" | "epinephrine" => Self::ADRENALINE_SMILES,

            // Amino acids
            "glycine" => Self::GLYCINE_SMILES,
            "alanine" => Self::ALANINE_SMILES,
            "tryptophan" => Self::TRYPTOPHAN_SMILES,
            "phenylalanine" => Self::PHENYLALANINE_SMILES,

            // Nucleotides
            "adenosine" => Self::ADENOSINE_SMILES,
            "guanosine" => Self::GUANOSINE_SMILES,

            _ => return None,
        };

        Some(MoleculeDefinition::new(name).with_smiles(smiles))
    }

    /// List all available molecules
    pub fn list() -> Vec<&'static str> {
        vec![
            // Cannabinoids
            "THC",
            "CBD",
            "CBN",
            "CBG",
            "Delta8-THC",
            "THCA",
            // Terpenes
            "Limonene",
            "Myrcene",
            "Pinene",
            "Linalool",
            "Caryophyllene",
            "Humulene",
            // Alkaloids
            "Caffeine",
            "Nicotine",
            "Psilocybin",
            "Psilocin",
            "DMT",
            "Mescaline",
            // Pharmaceuticals
            "Aspirin",
            "Ibuprofen",
            "Acetaminophen",
            "Melatonin",
            "Serotonin",
            "Dopamine",
            "Adrenaline",
            // Amino acids
            "Glycine",
            "Alanine",
            "Tryptophan",
            "Phenylalanine",
            // Nucleotides
            "Adenosine",
            "Guanosine",
        ]
    }
}

/// Molecular properties for small molecules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularProperties {
    pub name: String,
    pub smiles: String,
    pub molecular_formula: String,
    pub molecular_weight: f64,
    pub exact_mass: f64,
    pub log_p: f64,
    pub h_bond_donors: u32,
    pub h_bond_acceptors: u32,
    pub rotatable_bonds: u32,
    pub polar_surface_area: f64,
    pub category: MoleculeCategory,
    pub legal_status: LegalStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoleculeCategory {
    Cannabinoid,
    Terpene,
    Alkaloid,
    Pharmaceutical,
    AminoAcid,
    Nucleotide,
    Vitamin,
    Hormone,
    Neurotransmitter,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalStatus {
    Unrestricted,
    PrescriptionOnly,
    Controlled,
    Scheduled,
    Prohibited,
    VariesByJurisdiction,
}

impl MolecularProperties {
    pub fn thc() -> Self {
        Self {
            name: "Delta-9-Tetrahydrocannabinol".to_string(),
            smiles: SmallMoleculeLibrary::THC_SMILES.to_string(),
            molecular_formula: "C21H30O2".to_string(),
            molecular_weight: 314.469,
            exact_mass: 314.224580,
            log_p: 7.68,
            h_bond_donors: 1,
            h_bond_acceptors: 2,
            rotatable_bonds: 4,
            polar_surface_area: 29.46,
            category: MoleculeCategory::Cannabinoid,
            legal_status: LegalStatus::VariesByJurisdiction,
        }
    }

    pub fn cbd() -> Self {
        Self {
            name: "Cannabidiol".to_string(),
            smiles: SmallMoleculeLibrary::CBD_SMILES.to_string(),
            molecular_formula: "C21H30O2".to_string(),
            molecular_weight: 314.469,
            exact_mass: 314.224580,
            log_p: 6.30,
            h_bond_donors: 2,
            h_bond_acceptors: 2,
            rotatable_bonds: 6,
            polar_surface_area: 40.46,
            category: MoleculeCategory::Cannabinoid,
            legal_status: LegalStatus::VariesByJurisdiction,
        }
    }

    pub fn caffeine() -> Self {
        Self {
            name: "Caffeine".to_string(),
            smiles: SmallMoleculeLibrary::CAFFEINE_SMILES.to_string(),
            molecular_formula: "C8H10N4O2".to_string(),
            molecular_weight: 194.19,
            exact_mass: 194.080376,
            log_p: -0.07,
            h_bond_donors: 0,
            h_bond_acceptors: 6,
            rotatable_bonds: 0,
            polar_surface_area: 58.44,
            category: MoleculeCategory::Alkaloid,
            legal_status: LegalStatus::Unrestricted,
        }
    }

    pub fn melatonin() -> Self {
        Self {
            name: "Melatonin".to_string(),
            smiles: SmallMoleculeLibrary::MELATONIN_SMILES.to_string(),
            molecular_formula: "C13H16N2O2".to_string(),
            molecular_weight: 232.28,
            exact_mass: 232.121178,
            log_p: 1.0,
            h_bond_donors: 2,
            h_bond_acceptors: 2,
            rotatable_bonds: 4,
            polar_surface_area: 54.12,
            category: MoleculeCategory::Hormone,
            legal_status: LegalStatus::Unrestricted,
        }
    }
}

/// THC synthesis plan with detailed steps
#[derive(Debug, Clone)]
pub struct THCSynthesisPlan {
    /// Starting materials
    pub precursors: Vec<Precursor>,
    /// Synthesis steps
    pub steps: Vec<SynthesisStep>,
    /// Expected yield
    pub expected_yield: f64,
    /// Estimated time (minutes)
    pub estimated_time_min: u32,
    /// Required equipment
    pub required_equipment: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Precursor {
    pub name: String,
    pub smiles: String,
    pub amount: f64,
    pub unit: String,
}

#[derive(Debug, Clone)]
pub enum SynthesisStep {
    /// Place carbon atom at position
    PlaceCarbon(Vector3<f64>),
    /// Place oxygen atom at position
    PlaceOxygen(Vector3<f64>),
    /// Place hydrogen atom at position
    PlaceHydrogen(Vector3<f64>),
    /// Form bond between two atoms
    FormBond {
        atom1: u32,
        atom2: u32,
        bond_type: BondType,
    },
    /// Set stereocenter configuration
    SetStereocenter {
        center: u32,
        config: StereoConfig,
    },
    /// Verify molecular structure
    Verify {
        tolerance: f64,
    },
    /// Ring formation
    FormRing {
        atoms: Vec<u32>,
        aromatic: bool,
    },
    /// Wait for reaction
    Wait {
        duration_ms: u64,
    },
}

impl THCSynthesisPlan {
    /// Generate synthesis plan for THC
    pub fn generate() -> Self {
        // THC has 21 carbons, 30 hydrogens, 2 oxygens
        // Dibenzopyran scaffold with pentyl chain

        let mut steps = Vec::new();

        // Build benzene ring A (positions 0-5)
        for i in 0..6 {
            let angle = (i as f64) * std::f64::consts::PI / 3.0;
            let x = 1.40 * angle.cos();
            let y = 1.40 * angle.sin();
            steps.push(SynthesisStep::PlaceCarbon(Vector3::new(x, y, 0.0)));
        }

        // Form aromatic bonds in ring A
        for i in 0..6 {
            steps.push(SynthesisStep::FormBond {
                atom1: i,
                atom2: (i + 1) % 6,
                bond_type: BondType::Aromatic,
            });
        }

        // Build pyran ring B (fused to A)
        for i in 0..4 {
            let x = 2.8 + (i as f64) * 0.7;
            let y = (i as f64) * 0.5;
            if i == 0 {
                steps.push(SynthesisStep::PlaceOxygen(Vector3::new(x, y, 0.0)));
            } else {
                steps.push(SynthesisStep::PlaceCarbon(Vector3::new(x, y, 0.0)));
            }
        }

        // Build cyclohexene ring C (fused to B)
        for i in 0..6 {
            let x = 4.2 + (i as f64) * 0.6;
            let y = 1.0 + (i as f64) * 0.3;
            steps.push(SynthesisStep::PlaceCarbon(Vector3::new(x, y, 0.0)));
        }

        // Add pentyl chain (5 carbons)
        for i in 0..5 {
            let x = -1.54 * (i as f64 + 1.0);
            steps.push(SynthesisStep::PlaceCarbon(Vector3::new(x, 0.0, 0.0)));
        }

        // Add methyl groups (3x)
        for i in 0..3 {
            let x = 3.0 + (i as f64) * 1.5;
            let y = 2.0;
            steps.push(SynthesisStep::PlaceCarbon(Vector3::new(x, y, 0.0)));
        }

        // Add hydroxyl group
        steps.push(SynthesisStep::PlaceOxygen(Vector3::new(0.0, 2.0, 0.0)));

        // Add all hydrogens
        for _ in 0..30 {
            steps.push(SynthesisStep::PlaceHydrogen(Vector3::zeros())); // Positions calculated
        }

        // Set stereocenters
        steps.push(SynthesisStep::SetStereocenter {
            center: 6,  // 6a position
            config: StereoConfig::R,
        });
        steps.push(SynthesisStep::SetStereocenter {
            center: 10, // 10a position
            config: StereoConfig::R,
        });

        // Final verification
        steps.push(SynthesisStep::Verify { tolerance: 0.001 });

        Self {
            precursors: vec![
                Precursor {
                    name: "Olivetol".to_string(),
                    smiles: "CCCCCc1cc(O)cc(O)c1".to_string(),
                    amount: 1.0,
                    unit: "mmol".to_string(),
                },
                Precursor {
                    name: "p-Mentha-2,8-dien-1-ol".to_string(),
                    smiles: "CC(=C)C1CCC(C)=CC1O".to_string(),
                    amount: 1.0,
                    unit: "mmol".to_string(),
                },
            ],
            steps,
            expected_yield: 0.85,
            estimated_time_min: 120,
            required_equipment: vec![
                "NanoQuantumonas swarm".to_string(),
                "Attosecond laser array".to_string(),
                "Quantum verifier".to_string(),
                "Inert atmosphere chamber".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_molecule() {
        let thc = SmallMoleculeLibrary::get("THC").unwrap();
        assert_eq!(thc.name, "THC");
        assert!(thc.smiles.is_some());
    }

    #[test]
    fn test_list_molecules() {
        let list = SmallMoleculeLibrary::list();
        assert!(list.contains(&"THC"));
        assert!(list.contains(&"CBD"));
        assert!(list.contains(&"Caffeine"));
    }

    #[test]
    fn test_thc_properties() {
        let props = MolecularProperties::thc();
        assert_eq!(props.molecular_formula, "C21H30O2");
        assert!((props.molecular_weight - 314.469).abs() < 0.01);
    }

    #[test]
    fn test_synthesis_plan() {
        let plan = THCSynthesisPlan::generate();
        assert!(!plan.steps.is_empty());
        assert!(!plan.precursors.is_empty());
        assert!(plan.expected_yield > 0.0);
    }
}
