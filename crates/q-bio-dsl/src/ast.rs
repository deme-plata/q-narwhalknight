//! Abstract Syntax Tree for BioDSL
//!
//! Defines the AST nodes that represent parsed BioDSL programs.

use crate::types::*;
use serde::{Deserialize, Serialize};

/// Complete BioDSL program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioDSLProgram {
    pub definitions: Vec<Definition>,
    pub commands: Vec<Command>,
}

impl BioDSLProgram {
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            commands: Vec::new(),
        }
    }

    pub fn add_definition(&mut self, def: Definition) {
        self.definitions.push(def);
    }

    pub fn add_command(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    /// Get all molecule definitions
    pub fn molecules(&self) -> impl Iterator<Item = &MoleculeDefinition> {
        self.definitions.iter().filter_map(|d| match d {
            Definition::Molecule(m) => Some(m),
            _ => None,
        })
    }

    /// Get all genetic circuit definitions
    pub fn genetic_circuits(&self) -> impl Iterator<Item = &GeneticCircuitDefinition> {
        self.definitions.iter().filter_map(|d| match d {
            Definition::GeneticCircuit(g) => Some(g),
            _ => None,
        })
    }

    /// Get all protein definitions
    pub fn proteins(&self) -> impl Iterator<Item = &ProteinDefinition> {
        self.definitions.iter().filter_map(|d| match d {
            Definition::Protein(p) => Some(p),
            _ => None,
        })
    }
}

impl Default for BioDSLProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Definition {
    Molecule(MoleculeDefinition),
    GeneticCircuit(GeneticCircuitDefinition),
    Protein(ProteinDefinition),
    Library(LibraryDefinition),
}

/// Molecule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoleculeDefinition {
    pub name: String,
    pub smiles: Option<String>,
    pub scaffold: Option<ScaffoldDefinition>,
    pub substituents: Vec<SubstituentDefinition>,
    pub stereocenters: Vec<StereocenterDefinition>,
    pub synthesis_method: Option<SynthesisMethodSpec>,
    pub verification: Option<VerificationSpec>,
    pub properties: MoleculePropertySpec,
}

impl MoleculeDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            smiles: None,
            scaffold: None,
            substituents: Vec::new(),
            stereocenters: Vec::new(),
            synthesis_method: None,
            verification: None,
            properties: MoleculePropertySpec::default(),
        }
    }

    pub fn with_smiles(mut self, smiles: &str) -> Self {
        self.smiles = Some(smiles.to_string());
        self
    }
}

/// Molecular scaffold definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaffoldDefinition {
    pub name: String,
    pub rings: Vec<RingDefinition>,
}

/// Ring definition within scaffold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingDefinition {
    pub name: String,
    pub ring_type: RingTypeSpec,
    pub position: Option<PositionSpec>,
    pub fused_to: Option<FusionSpec>,
}

/// Ring type specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RingTypeSpec {
    Benzene,
    Pyran,
    Cyclohexene,
    Cyclohexane,
    Cyclopentane,
    Pyridine,
    Pyrrole,
    Furan,
    Thiophene,
    Imidazole,
    Custom { size: usize, aromatic: bool },
}

/// Position specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSpec {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl PositionSpec {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Ring fusion specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSpec {
    pub target_ring: String,
    pub positions: Vec<usize>,
}

/// Substituent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstituentDefinition {
    pub group: FunctionalGroup,
    pub position: SubstituentPosition,
    pub count: usize,
}

/// Functional group types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionalGroup {
    Hydroxyl,
    Methyl,
    Ethyl,
    Propyl,
    Isopropyl,
    Butyl,
    Pentyl,
    Amino,
    Carboxyl,
    Carbonyl,
    Aldehyde,
    Ester,
    Ether,
    Halogen(Element),
    Nitro,
    Sulfhydryl,
    Phosphate,
    Custom(String),
}

/// Substituent position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubstituentPosition {
    RingPosition { ring: String, position: usize },
    AtomId(AtomId),
    Named(String),
    Multiple(Vec<SubstituentPosition>),
}

/// Stereocenter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StereocenterDefinition {
    pub name: String,
    pub config: StereoConfig,
}

/// Synthesis method specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisMethodSpec {
    pub robot_type: String,
    pub method: String,
    pub parameters: Vec<(String, ParameterValue)>,
}

/// Verification specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSpec {
    pub method: String,
    pub tolerance: f64,
    pub parameters: Vec<(String, ParameterValue)>,
}

/// Parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    List(Vec<ParameterValue>),
}

/// Molecule property specification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoleculePropertySpec {
    pub target_purity: Option<f64>,
    pub target_yield: Option<f64>,
    pub chirality_required: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENETIC CIRCUIT AST
// ═══════════════════════════════════════════════════════════════════════════════

/// Genetic circuit definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticCircuitDefinition {
    pub name: String,
    pub promoters: Vec<PromoterDefinition>,
    pub genes: Vec<GeneDefinition>,
    pub terminators: Vec<TerminatorDefinition>,
    pub interactions: Vec<GeneticInteraction>,
    pub inputs: Vec<CircuitInput>,
    pub outputs: Vec<CircuitOutput>,
    pub safety: Option<SafetyFeatures>,
}

impl GeneticCircuitDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            promoters: Vec::new(),
            genes: Vec::new(),
            terminators: Vec::new(),
            interactions: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            safety: None,
        }
    }
}

/// Promoter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoterDefinition {
    pub name: String,
    pub binding_sites: Vec<String>,
    pub strength: f64,
    pub constitutive: bool,
    pub sequence: Option<String>,
}

impl PromoterDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            binding_sites: Vec::new(),
            strength: 1.0,
            constitutive: false,
            sequence: None,
        }
    }
}

/// Gene definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneDefinition {
    pub name: String,
    pub promoter: String,
    pub product: String,
    pub degradation_time: Option<DurationSpec>,
    pub sequence: Option<String>,
    pub codon_optimized_for: Option<String>,
}

impl GeneDefinition {
    pub fn new(name: &str, promoter: &str, product: &str) -> Self {
        Self {
            name: name.to_string(),
            promoter: promoter.to_string(),
            product: product.to_string(),
            degradation_time: None,
            sequence: None,
            codon_optimized_for: None,
        }
    }
}

/// Terminator definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminatorDefinition {
    pub name: String,
    pub efficiency: f64,
    pub bidirectional: bool,
    pub sequence: Option<String>,
}

/// Duration specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationSpec {
    pub value: f64,
    pub unit: TimeUnit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

impl DurationSpec {
    pub fn minutes(value: f64) -> Self {
        Self {
            value,
            unit: TimeUnit::Minutes,
        }
    }

    pub fn to_seconds(&self) -> f64 {
        match self.unit {
            TimeUnit::Seconds => self.value,
            TimeUnit::Minutes => self.value * 60.0,
            TimeUnit::Hours => self.value * 3600.0,
            TimeUnit::Days => self.value * 86400.0,
        }
    }
}

/// Genetic interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneticInteraction {
    Repression {
        repressor: String,
        target: String,
        strength: f64,
    },
    Activation {
        activator: String,
        target: String,
        strength: f64,
    },
    Fusion {
        gene1: String,
        gene2: String,
    },
}

/// Circuit input signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInput {
    pub name: String,
    pub action: InputAction,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputAction {
    SwitchesOff,
    SwitchesOn,
    Modulates { factor: f64 },
}

/// Circuit output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitOutput {
    pub name: String,
    pub reporter_type: ReporterType,
    pub fused_to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReporterType {
    GFP,
    RFP,
    YFP,
    CFP,
    BFP,
    Luciferase,
    LacZ,
    Custom(String),
}

/// Safety features for genetic circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyFeatures {
    pub auxotrophy: Vec<String>,
    pub kill_switches: Vec<KillSwitchSpec>,
    pub generation_limit: Option<u32>,
    pub genetic_firewall: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchSpec {
    pub trigger: KillTriggerSpec,
    pub toxin: String,
    pub response_time: DurationSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillTriggerSpec {
    NutrientDependent(String),
    LightSensitive,
    TemperatureSensitive { min: f64, max: f64 },
    ChemicalInducible(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROTEIN AST
// ═══════════════════════════════════════════════════════════════════════════════

/// Protein definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinDefinition {
    pub name: String,
    pub sequence: String,
    pub structure: Option<ProteinStructureSpec>,
    pub optimization: Option<OptimizationSpec>,
    pub folding_method: Option<FoldingMethodSpec>,
}

impl ProteinDefinition {
    pub fn new(name: &str, sequence: &str) -> Self {
        Self {
            name: name.to_string(),
            sequence: sequence.to_string(),
            structure: None,
            optimization: None,
            folding_method: None,
        }
    }
}

/// Protein structure specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinStructureSpec {
    pub domains: Vec<DomainSpec>,
    pub active_sites: Vec<ActiveSiteSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSpec {
    pub name: String,
    pub domain_type: DomainType,
    pub residue_range: (usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainType {
    AlphaHelix,
    BetaSheet,
    Loop,
    AlphaHelixBundle,
    BetaBarrel,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSiteSpec {
    pub catalytic_residues: Vec<String>,
    pub binding_pocket_type: String,
}

/// Optimization specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSpec {
    pub stability: OptimizationGoal,
    pub solubility: OptimizationConstraint,
    pub expression_host: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationGoal {
    Maximize,
    Minimize,
    Target(f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationConstraint {
    GreaterThan(f64),
    LessThan(f64),
    Between(f64, f64),
}

/// Folding method specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingMethodSpec {
    pub primary_method: String,
    pub fallback_method: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIBRARY AST
// ═══════════════════════════════════════════════════════════════════════════════

/// Library definition for combinatorial chemistry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryDefinition {
    pub name: String,
    pub scaffold: String,
    pub variations: Vec<VariationSpec>,
    pub generation_mode: GenerationMode,
    pub quantity_each: Quantity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationSpec {
    pub position: String,
    pub substituents: Vec<FunctionalGroup>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationMode {
    Combinatorial,
    Sequential,
    Random { count: usize },
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMANDS
// ═══════════════════════════════════════════════════════════════════════════════

/// Executable command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    Synthesize(SynthesizeCommand),
    CompileToDNA(CompileToDNACommand),
    Transform(TransformCommand),
    Verify(VerifyCommand),
    Print(PrintCommand),
}

/// Synthesize command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeCommand {
    pub target: String,
    pub quantity: Quantity,
    pub purity: Option<f64>,
    pub robot_override: Option<String>,
}

impl SynthesizeCommand {
    pub fn new(target: &str, quantity: Quantity) -> Self {
        Self {
            target: target.to_string(),
            quantity,
            purity: None,
            robot_override: None,
        }
    }
}

/// Compile to DNA command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileToDNACommand {
    pub circuit: String,
    pub output_variable: Option<String>,
}

/// Transform command (insert DNA into host)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformCommand {
    pub host: String,
    pub plasmid: String,
    pub method: TransformMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformMethod {
    HeatShock,
    Electroporation,
    Chemical,
    NanoQuantumonas, // Quantum-assisted delivery
}

/// Verify command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyCommand {
    pub target: String,
    pub method: String,
    pub tolerance: f64,
}

/// Print/bio-print command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrintCommand {
    pub target: String,
    pub printer_type: PrinterType,
    pub parameters: Vec<(String, ParameterValue)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrinterType {
    BioPrint3D,
    Inkjet,
    Extrusion,
    Stereolithography,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_definition() {
        let mol = MoleculeDefinition::new("THC").with_smiles("CCCCCC1=CC...");
        assert_eq!(mol.name, "THC");
        assert!(mol.smiles.is_some());
    }

    #[test]
    fn test_genetic_circuit() {
        let mut circuit = GeneticCircuitDefinition::new("ToggleSwitch");
        circuit.promoters.push(PromoterDefinition::new("pTet"));
        circuit
            .genes
            .push(GeneDefinition::new("lacI", "pTet", "LacI"));

        assert_eq!(circuit.name, "ToggleSwitch");
        assert_eq!(circuit.promoters.len(), 1);
        assert_eq!(circuit.genes.len(), 1);
    }

    #[test]
    fn test_protein_definition() {
        let protein = ProteinDefinition::new("TestProtein", "MVLSPADKTNVK");
        assert_eq!(protein.name, "TestProtein");
        assert_eq!(protein.sequence, "MVLSPADKTNVK");
    }
}
