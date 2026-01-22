//! Genetic Circuit Components and Templates
//!
//! Pre-defined genetic circuit components including promoters,
//! genes, terminators, and complete circuit templates.

use crate::ast::*;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Registry of standard biological parts
pub struct PartsRegistry {
    promoters: HashMap<String, PromoterPart>,
    ribosome_binding_sites: HashMap<String, RBSPart>,
    coding_sequences: HashMap<String, CDSPart>,
    terminators: HashMap<String, TerminatorPart>,
}

impl PartsRegistry {
    /// Create registry with standard iGEM/BioBricks parts
    pub fn new() -> Self {
        let mut registry = Self {
            promoters: HashMap::new(),
            ribosome_binding_sites: HashMap::new(),
            coding_sequences: HashMap::new(),
            terminators: HashMap::new(),
        };

        registry.load_standard_parts();
        registry
    }

    fn load_standard_parts(&mut self) {
        // Standard promoters
        self.promoters.insert(
            "pTet".to_string(),
            PromoterPart {
                name: "pTet".to_string(),
                biobrick_id: Some("BBa_R0040".to_string()),
                sequence: "TCCCTATCAGTGATAGAGATTGACATCCCTATCAGTGATAGAGATACTGAGCAC".to_string(),
                strength: 0.85,
                repressor: Some("TetR".to_string()),
                inducer: Some("aTc".to_string()),
            },
        );

        self.promoters.insert(
            "pLac".to_string(),
            PromoterPart {
                name: "pLac".to_string(),
                biobrick_id: Some("BBa_R0010".to_string()),
                sequence: "CAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATTTCACACA".to_string(),
                strength: 0.80,
                repressor: Some("LacI".to_string()),
                inducer: Some("IPTG".to_string()),
            },
        );

        self.promoters.insert(
            "pLambda".to_string(),
            PromoterPart {
                name: "pLambda".to_string(),
                biobrick_id: Some("BBa_R0051".to_string()),
                sequence: "TAACACCGTGCGTGTTGACTATTTTACCTCTGGCGGTGATAATGGTTGC".to_string(),
                strength: 0.90,
                repressor: Some("CI".to_string()),
                inducer: None,
            },
        );

        self.promoters.insert(
            "pBAD".to_string(),
            PromoterPart {
                name: "pBAD".to_string(),
                biobrick_id: Some("BBa_I0500".to_string()),
                sequence: "ACTTTTCATACTCCCGCCATTCAGAGAAGAAACCAATTGTCCATATTGCATCAGACATTGCCGTCACTGCGTCTTTTACTGGCTCTTCTCGCTAACCAAACCGGTAACCCCGCTTATTAAAAGCATTCTGTAACAAAGCGGGACCAAAGCCATGACAAAAACGCGTAACAAAAGTGTCTATAATCACGGCAGAAAAGTCCACATTGATTATTTGCACGGCGTCACACTTTGCTATGCCATAGCATTTTTATCCATAAGATTAGCGGATCCTACCTGACGCTTTTTATCGCAACTCTCTACTGTTTCTCCATACCCGTTTTTTTGGGCTAGC".to_string(),
                strength: 0.75,
                repressor: Some("AraC".to_string()),
                inducer: Some("Arabinose".to_string()),
            },
        );

        // Constitutive promoters
        self.promoters.insert(
            "J23100".to_string(),
            PromoterPart {
                name: "J23100".to_string(),
                biobrick_id: Some("BBa_J23100".to_string()),
                sequence: "TTGACGGCTAGCTCAGTCCTAGGTACAGTGCTAGC".to_string(),
                strength: 1.0, // Reference promoter
                repressor: None,
                inducer: None,
            },
        );

        self.promoters.insert(
            "J23119".to_string(),
            PromoterPart {
                name: "J23119".to_string(),
                biobrick_id: Some("BBa_J23119".to_string()),
                sequence: "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC".to_string(),
                strength: 0.70,
                repressor: None,
                inducer: None,
            },
        );

        // Ribosome binding sites
        self.ribosome_binding_sites.insert(
            "B0034".to_string(),
            RBSPart {
                name: "B0034".to_string(),
                biobrick_id: Some("BBa_B0034".to_string()),
                sequence: "AAAGAGGAGAAA".to_string(),
                strength: 1.0,
            },
        );

        self.ribosome_binding_sites.insert(
            "B0030".to_string(),
            RBSPart {
                name: "B0030".to_string(),
                biobrick_id: Some("BBa_B0030".to_string()),
                sequence: "ATTAAAGAGGAGAAA".to_string(),
                strength: 0.60,
            },
        );

        self.ribosome_binding_sites.insert(
            "B0032".to_string(),
            RBSPart {
                name: "B0032".to_string(),
                biobrick_id: Some("BBa_B0032".to_string()),
                sequence: "TCACACAGGAAAG".to_string(),
                strength: 0.30,
            },
        );

        // Coding sequences (repressors and reporters)
        self.coding_sequences.insert(
            "lacI".to_string(),
            CDSPart {
                name: "lacI".to_string(),
                biobrick_id: Some("BBa_C0012".to_string()),
                sequence: "ATGGTGAATGTGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTGAACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATTCCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCCACCTCCAGTCTGGCCCTGCACGCGCCGTCGCAAATTGTCGCGGCGATTAAATCTCGCGCCGATCAACTGGGTGCCAGCGTGGTGGTGTCGATGGTAGAACGAAGCGGCGTCGAAGCCTGTAAAGCGGCGGTGCACAATCTTCTCGCGCAACGCGTCAGTGGGCTGATCATTAACTATCCGCTGGATGACCAGGATGCCATTGCTGTGGAAGCTGCCTGCACTAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACACCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTGGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTGCGTCTGGCTGGCTGGCATAAATATCTCACTCGCAATCAAATTCAGCCGATAGCGGAACGGGAAGGCGACTGGAGTGCCATGTCCGGTTTTCAACAAACCATGCAAATGCTGAATGAGGGCATCGTTCCCACTGCGATGCTGGTTGCCAACGATCAGATGGCGCTGGGCGCAATGCGCGCCATTACCGAGTCCGGGCTGCGCGTTGGTGCGGATATCTCGGTAGTGGGATACGACGATACCGAAGACAGCTCATGTTATATCCCGCCGTTAACCACCATCAAACAGGATTTTCGCCTGCTGGGGCAAACCAGCGTGGACCGCTTGCTGCAACTCTCTCAGGGCCAGGCGGTGAAGGGCAATCAGCTGTTGCCCGTCTCACTGGTGAAAAGAAAAACCACCCTGGCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGA".to_string(),
                protein: "LacI".to_string(),
                length_aa: 360,
            },
        );

        self.coding_sequences.insert(
            "tetR".to_string(),
            CDSPart {
                name: "tetR".to_string(),
                biobrick_id: Some("BBa_C0040".to_string()),
                sequence: "ATGTCCAGATTAGATAAAAGTAAAGTGATTAACAGCGCATTAGAGCTGCTTAATGAGGTCGGAATCGAAGGTTTAACAACCCGTAAACTCGCCCAGAAGCTAGGTGTAGAGCAGCCTACATTGTATTGGCATGTAAAAAATAAGCGGGCTTTGCTCGACGCCTTAGCCATTGAGATGTTAGATAGGCACCATACTCACTTTTGCCCTTTAGAAGGGGAAAGCTGGCAAGATTTTTTACGTAATAACGCTAAAAGTTTTAGATGTGCTTTACTAAGTCATCGCGATGGAGCAAAAGTACATTTAGGTACACGGCCTACAGAAAAACAGTATGAAACTCTCGAAAATCAATTAGCCTTTTTATGCCAACAAGGTTTTTCACTAGAGAATGCATTATATGCACTCAGCGCTGTGGGGCATTTTACTTTAGGTTGCGTATTGGAAGATCAAGAGCATCAAGTCGCTAAAGAAGAAAGGGAAACACCTACTACTGATAGTATGCCGCCATTATTACGACAAGCTATCGAATTATTTGATCACCAAGGTGCAGAGCCAGCCTTCTTATTCGGCCTTGAATTGATCATATGCGGATTAGAAAAACAACTTAAATGTGAAAGTGGGTCCGCTGCAAACGACGAAAACTACGCTTTAGTAGCTTAATAA".to_string(),
                protein: "TetR".to_string(),
                length_aa: 207,
            },
        );

        self.coding_sequences.insert(
            "cI".to_string(),
            CDSPart {
                name: "cI".to_string(),
                biobrick_id: Some("BBa_C0051".to_string()),
                sequence: "ATGAGCACAAAAAAGAAACCATTAACACAAGAGCAGCTTGAGGACGCACGTCGCCTTAAAGCAATTTATGAAAAAAAGAAAAATGAACTTGGCTTATCCCAGGAATCTGTCGCAGACAAGATGGGGATGGGGCAGTCAGGCGTTGGTGCTTTATTTAATGGCATCAATGCATTAAATGCTTATAACGCCGCATTGCTTGCAAAAATTCTCAAAGTTAGCGTTGAAGAATTTAGCCCTTCAATCGCCAGAGAAATCTACGAGATGTATGAAGCGGTTAGTATGCAGCCGTCACTTAGAAGTGAGTATGAGTACCCTGTTTTTTCTCATGTTCAGGCAGGGATGTTCTCACCTGAGCTTAGAACCTTTACCAAAGGTGATGCGGAGAGATGGGTAAGCACAACCAAAAAAGCCAGTGATTCTGCATTCTGGCTTGAGGTTGAAGGTAATTCCATGACCGCACCAACAGGCTCCAAGCCAAGCTTTCCTGACGGAATGTTAATTCTCGTTGACCCTGAGCAGGCTGTTGAGCCAGGTGATTTCTGCATAGCCAGACTTGGGGGTGATGAGTTTACCTTCAAGAAACTGATCAGGGATAGCGGTCAGGTGTTTTTACAACCACTAAACCCACAGTACCCAATGATCCCATGCAATGAGAGTTGTTCCGTTGTGGGGAAAGTTATCGCTAGTCAGTGGCCTGAAGAGACGTTTGGC".to_string(),
                protein: "CI".to_string(),
                length_aa: 236,
            },
        );

        // Reporter genes
        self.coding_sequences.insert(
            "GFP".to_string(),
            CDSPart {
                name: "GFP".to_string(),
                biobrick_id: Some("BBa_E0040".to_string()),
                sequence: "ATGCGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCGGTTATGGTGTTCAATGCTTTGCGAGATACCCAGATCATATGAAACAGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACTATATTTTTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTTGGACACAAATTGGAATACAACTATAACTCACACAATGTATACATCATGGCAGACAAACAAAAGAATGGAATCAAAGTTAACTTCAAAATTAGACACAACATTGAAGATGGAAGCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCCAACGAAAAGAGAGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATGGATGAACTATACAAATAA".to_string(),
                protein: "GFP".to_string(),
                length_aa: 239,
            },
        );

        self.coding_sequences.insert(
            "RFP".to_string(),
            CDSPart {
                name: "RFP".to_string(),
                biobrick_id: Some("BBa_E1010".to_string()),
                sequence: "ATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAA".to_string(),
                protein: "RFP".to_string(),
                length_aa: 225,
            },
        );

        // Terminators
        self.terminators.insert(
            "B0015".to_string(),
            TerminatorPart {
                name: "B0015".to_string(),
                biobrick_id: Some("BBa_B0015".to_string()),
                sequence: "CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATA".to_string(),
                efficiency: 0.99,
                bidirectional: true,
            },
        );

        self.terminators.insert(
            "B0010".to_string(),
            TerminatorPart {
                name: "B0010".to_string(),
                biobrick_id: Some("BBa_B0010".to_string()),
                sequence: "CCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTC".to_string(),
                efficiency: 0.98,
                bidirectional: false,
            },
        );
    }

    /// Get promoter by name
    pub fn get_promoter(&self, name: &str) -> Option<&PromoterPart> {
        self.promoters.get(name)
    }

    /// Get RBS by name
    pub fn get_rbs(&self, name: &str) -> Option<&RBSPart> {
        self.ribosome_binding_sites.get(name)
    }

    /// Get coding sequence by name
    pub fn get_cds(&self, name: &str) -> Option<&CDSPart> {
        self.coding_sequences.get(name)
    }

    /// Get terminator by name
    pub fn get_terminator(&self, name: &str) -> Option<&TerminatorPart> {
        self.terminators.get(name)
    }
}

impl Default for PartsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Promoter part definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoterPart {
    pub name: String,
    pub biobrick_id: Option<String>,
    pub sequence: String,
    pub strength: f64,
    pub repressor: Option<String>,
    pub inducer: Option<String>,
}

/// Ribosome binding site part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBSPart {
    pub name: String,
    pub biobrick_id: Option<String>,
    pub sequence: String,
    pub strength: f64,
}

/// Coding sequence part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDSPart {
    pub name: String,
    pub biobrick_id: Option<String>,
    pub sequence: String,
    pub protein: String,
    pub length_aa: usize,
}

/// Terminator part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminatorPart {
    pub name: String,
    pub biobrick_id: Option<String>,
    pub sequence: String,
    pub efficiency: f64,
    pub bidirectional: bool,
}

/// Genetic circuit compiler
pub struct GeneticCircuitCompiler {
    registry: PartsRegistry,
}

impl GeneticCircuitCompiler {
    pub fn new() -> Self {
        Self {
            registry: PartsRegistry::new(),
        }
    }

    /// Compile genetic circuit to DNA sequence
    pub fn compile(&self, circuit: &GeneticCircuitDefinition) -> Result<DNASequence, String> {
        let mut sequence = DNASequence::new();

        for gene in &circuit.genes {
            // Add promoter
            if let Some(promoter) = self.registry.get_promoter(&gene.promoter) {
                if let Some(seq) = DNASequence::from_string(&promoter.sequence) {
                    sequence.append(&seq);
                }
            }

            // Add default RBS
            if let Some(rbs) = self.registry.get_rbs("B0034") {
                if let Some(seq) = DNASequence::from_string(&rbs.sequence) {
                    sequence.append(&seq);
                }
            }

            // Add coding sequence
            if let Some(cds) = self.registry.get_cds(&gene.name) {
                if let Some(seq) = DNASequence::from_string(&cds.sequence) {
                    sequence.append(&seq);
                }
            }

            // Add terminator
            if let Some(term) = self.registry.get_terminator("B0015") {
                if let Some(seq) = DNASequence::from_string(&term.sequence) {
                    sequence.append(&seq);
                }
            }
        }

        Ok(sequence)
    }

    /// Create toggle switch circuit
    pub fn toggle_switch() -> GeneticCircuitDefinition {
        let mut circuit = GeneticCircuitDefinition::new("ToggleSwitch");

        circuit.promoters.push(PromoterDefinition {
            name: "pTet".to_string(),
            binding_sites: vec!["tet_operator".to_string()],
            strength: 0.85,
            constitutive: false,
            sequence: None,
        });

        circuit.promoters.push(PromoterDefinition {
            name: "pLac".to_string(),
            binding_sites: vec!["lac_operator".to_string()],
            strength: 0.80,
            constitutive: false,
            sequence: None,
        });

        let mut gene1 = GeneDefinition::new("lacI", "pTet", "LacI");
        gene1.degradation_time = Some(DurationSpec::minutes(30.0));
        circuit.genes.push(gene1);

        let mut gene2 = GeneDefinition::new("tetR", "pLac", "TetR");
        gene2.degradation_time = Some(DurationSpec::minutes(45.0));
        circuit.genes.push(gene2);

        circuit.interactions.push(GeneticInteraction::Repression {
            repressor: "lacI".to_string(),
            target: "pLac".to_string(),
            strength: 1.0,
        });

        circuit.interactions.push(GeneticInteraction::Repression {
            repressor: "tetR".to_string(),
            target: "pTet".to_string(),
            strength: 1.0,
        });

        circuit.inputs.push(CircuitInput {
            name: "IPTG".to_string(),
            action: InputAction::SwitchesOff,
            target: "lacI".to_string(),
        });

        circuit.inputs.push(CircuitInput {
            name: "aTc".to_string(),
            action: InputAction::SwitchesOff,
            target: "tetR".to_string(),
        });

        circuit.outputs.push(CircuitOutput {
            name: "GFP".to_string(),
            reporter_type: ReporterType::GFP,
            fused_to: "lacI".to_string(),
        });

        circuit.outputs.push(CircuitOutput {
            name: "RFP".to_string(),
            reporter_type: ReporterType::RFP,
            fused_to: "tetR".to_string(),
        });

        circuit
    }

    /// Create repressilator circuit (3-gene oscillator)
    pub fn repressilator() -> GeneticCircuitDefinition {
        let mut circuit = GeneticCircuitDefinition::new("Repressilator");

        // Three promoters
        circuit.promoters.push(PromoterDefinition {
            name: "pTet".to_string(),
            binding_sites: vec!["tet_operator".to_string()],
            strength: 0.85,
            constitutive: false,
            sequence: None,
        });

        circuit.promoters.push(PromoterDefinition {
            name: "pLac".to_string(),
            binding_sites: vec!["lac_operator".to_string()],
            strength: 0.80,
            constitutive: false,
            sequence: None,
        });

        circuit.promoters.push(PromoterDefinition {
            name: "pLambda".to_string(),
            binding_sites: vec!["lambda_operator".to_string()],
            strength: 0.90,
            constitutive: false,
            sequence: None,
        });

        // Three genes in circular repression
        circuit.genes.push(GeneDefinition::new("tetR", "pLambda", "TetR"));
        circuit.genes.push(GeneDefinition::new("lacI", "pTet", "LacI"));
        circuit.genes.push(GeneDefinition::new("cI", "pLac", "CI"));

        // Circular repression: TetR -| LacI -| CI -| TetR
        circuit.interactions.push(GeneticInteraction::Repression {
            repressor: "tetR".to_string(),
            target: "pTet".to_string(),
            strength: 1.0,
        });

        circuit.interactions.push(GeneticInteraction::Repression {
            repressor: "lacI".to_string(),
            target: "pLac".to_string(),
            strength: 1.0,
        });

        circuit.interactions.push(GeneticInteraction::Repression {
            repressor: "cI".to_string(),
            target: "pLambda".to_string(),
            strength: 1.0,
        });

        // GFP reporter fused to TetR
        circuit.outputs.push(CircuitOutput {
            name: "GFP".to_string(),
            reporter_type: ReporterType::GFP,
            fused_to: "tetR".to_string(),
        });

        circuit
    }

    /// Create AND gate circuit
    pub fn and_gate(input1: &str, input2: &str, output: &str) -> GeneticCircuitDefinition {
        let mut circuit = GeneticCircuitDefinition::new(&format!("AND_{}_{}", input1, input2));

        // AND gate uses split intein mechanism or T7 polymerase fragments
        circuit.promoters.push(PromoterDefinition {
            name: format!("p{}", input1),
            binding_sites: vec![format!("{}_operator", input1.to_lowercase())],
            strength: 0.80,
            constitutive: false,
            sequence: None,
        });

        circuit.promoters.push(PromoterDefinition {
            name: format!("p{}", input2),
            binding_sites: vec![format!("{}_operator", input2.to_lowercase())],
            strength: 0.80,
            constitutive: false,
            sequence: None,
        });

        // Split T7 RNAP fragments
        circuit.genes.push(GeneDefinition::new(
            "T7_N",
            &format!("p{}", input1),
            "T7_RNAP_N_terminal",
        ));

        circuit.genes.push(GeneDefinition::new(
            "T7_C",
            &format!("p{}", input2),
            "T7_RNAP_C_terminal",
        ));

        // Output under T7 promoter
        circuit.genes.push(GeneDefinition::new(output, "pT7", output));

        circuit.inputs.push(CircuitInput {
            name: input1.to_string(),
            action: InputAction::SwitchesOn,
            target: "T7_N".to_string(),
        });

        circuit.inputs.push(CircuitInput {
            name: input2.to_string(),
            action: InputAction::SwitchesOn,
            target: "T7_C".to_string(),
        });

        circuit.outputs.push(CircuitOutput {
            name: output.to_string(),
            reporter_type: ReporterType::Custom(output.to_string()),
            fused_to: "".to_string(),
        });

        circuit
    }
}

impl Default for GeneticCircuitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parts_registry() {
        let registry = PartsRegistry::new();

        assert!(registry.get_promoter("pTet").is_some());
        assert!(registry.get_promoter("pLac").is_some());
        assert!(registry.get_rbs("B0034").is_some());
        assert!(registry.get_cds("GFP").is_some());
        assert!(registry.get_terminator("B0015").is_some());
    }

    #[test]
    fn test_toggle_switch() {
        let circuit = GeneticCircuitCompiler::toggle_switch();

        assert_eq!(circuit.name, "ToggleSwitch");
        assert_eq!(circuit.promoters.len(), 2);
        assert_eq!(circuit.genes.len(), 2);
        assert_eq!(circuit.interactions.len(), 2);
        assert_eq!(circuit.inputs.len(), 2);
        assert_eq!(circuit.outputs.len(), 2);
    }

    #[test]
    fn test_repressilator() {
        let circuit = GeneticCircuitCompiler::repressilator();

        assert_eq!(circuit.name, "Repressilator");
        assert_eq!(circuit.genes.len(), 3);
        assert_eq!(circuit.interactions.len(), 3);
    }

    #[test]
    fn test_compile_circuit() {
        let compiler = GeneticCircuitCompiler::new();
        let circuit = GeneticCircuitCompiler::toggle_switch();

        let dna = compiler.compile(&circuit).unwrap();
        assert!(!dna.is_empty());
    }
}
