//! Molecular Instruction Set (MIS)
//!
//! Low-level instructions for controlling quantum water robots
//! to perform atomic-scale molecular assembly.

use crate::types::*;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Molecular instruction for robot execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MolecularInstruction {
    /// Place a single atom at precise coordinates
    PlaceAtom {
        instruction_id: u64,
        element: Element,
        position: Vector3<f64>,
        robot_type: RobotType,
        quantum_precision: bool,
    },

    /// Form a chemical bond between two atoms
    FormBond {
        instruction_id: u64,
        atom1_id: AtomId,
        atom2_id: AtomId,
        bond_type: BondType,
        use_laser: bool,
    },

    /// Break an existing bond
    BreakBond {
        instruction_id: u64,
        atom1_id: AtomId,
        atom2_id: AtomId,
    },

    /// Move atom to new position
    MoveAtom {
        instruction_id: u64,
        atom_id: AtomId,
        new_position: Vector3<f64>,
    },

    /// Rotate molecule or fragment
    RotateMolecule {
        instruction_id: u64,
        atom_ids: Vec<AtomId>,
        axis: Vector3<f64>,
        angle_radians: f64,
    },

    /// Apply attosecond laser pulse
    LaserPulse {
        instruction_id: u64,
        target: Vector3<f64>,
        energy_ev: f64,
        duration_attoseconds: u64,
        pulse_type: LaserPulseType,
    },

    /// Verify molecular structure via quantum tomography
    VerifyStructure {
        instruction_id: u64,
        molecule_id: String,
        expected_atoms: usize,
        expected_bonds: usize,
        tolerance: f64,
    },

    /// Verify stereochemistry
    VerifyStereocenter {
        instruction_id: u64,
        center_atom: AtomId,
        expected_config: StereoConfig,
    },

    /// Assemble ring structure
    AssembleRing {
        instruction_id: u64,
        ring_type: RingAssemblyType,
        center: Vector3<f64>,
        orientation: Vector3<f64>,
    },

    /// Build scaffold from template
    BuildScaffold {
        instruction_id: u64,
        scaffold_name: String,
        position: Vector3<f64>,
        scale: f64,
    },

    /// Attach substituent group
    AttachSubstituent {
        instruction_id: u64,
        target_atom: AtomId,
        substituent: SubstituentType,
        orientation: Vector3<f64>,
    },

    /// Synthesize DNA sequence
    SynthesizeDNA {
        instruction_id: u64,
        sequence: String,
        start_position: Vector3<f64>,
    },

    /// Fold protein with quantum assistance
    AssistProteinFolding {
        instruction_id: u64,
        sequence: String,
        target_structure_id: Option<String>,
    },

    /// Build Metal-Organic Framework
    BuildMOF {
        instruction_id: u64,
        metal: Element,
        linker: String,
        topology: String,
        size: MOFSize,
    },

    /// Build Covalent-Organic Framework
    BuildCOF {
        instruction_id: u64,
        linkage_type: String,
        dimension: COFDimension,
        size: f64,
    },

    /// Build Zeolitic Imidazolate Framework
    BuildZIF {
        instruction_id: u64,
        metal: Element,
        imidazolate: String,
        topology: String,
    },

    /// Wait for process completion
    Wait {
        instruction_id: u64,
        duration_ms: u64,
    },

    /// Checkpoint for quantum state verification
    Checkpoint {
        instruction_id: u64,
        description: String,
    },

    /// Parallel execution of multiple instructions
    Parallel {
        instruction_id: u64,
        instructions: Vec<MolecularInstruction>,
    },

    /// Conditional execution
    Conditional {
        instruction_id: u64,
        condition: VerificationCondition,
        if_true: Vec<MolecularInstruction>,
        if_false: Vec<MolecularInstruction>,
    },

    /// Loop execution
    Loop {
        instruction_id: u64,
        count: usize,
        instructions: Vec<MolecularInstruction>,
    },
}

/// Robot type for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RobotType {
    NanoQuantumonas,
    QuantumJellyfish,
    TunnelingOctopus,
    EntangledDolphin,
    WaveParticleWhale,
    SuperpositionSeahorse,
    SchoolingRobotichthys,
    CyberCetus,
    Any,
}

impl RobotType {
    pub fn best_for_operation(op: &MolecularInstruction) -> Self {
        match op {
            MolecularInstruction::PlaceAtom { .. } => RobotType::NanoQuantumonas,
            MolecularInstruction::FormBond { .. } => RobotType::NanoQuantumonas,
            MolecularInstruction::BuildMOF { .. } => RobotType::QuantumJellyfish,
            MolecularInstruction::BuildCOF { .. } => RobotType::EntangledDolphin,
            MolecularInstruction::BuildZIF { .. } => RobotType::TunnelingOctopus,
            MolecularInstruction::AssistProteinFolding { .. } => RobotType::CyberCetus,
            MolecularInstruction::Parallel { .. } => RobotType::SchoolingRobotichthys,
            _ => RobotType::NanoQuantumonas,
        }
    }
}

/// Laser pulse type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaserPulseType {
    /// Bond formation assistance
    BondFormation,
    /// Bond breaking
    BondBreaking,
    /// Excitation for imaging
    Imaging,
    /// Ionization
    Ionization,
    /// Coherence refresh
    CoherenceRefresh,
}

/// Ring assembly type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RingAssemblyType {
    Benzene,
    Cyclohexane,
    Cyclopentane,
    Pyridine,
    Pyrrole,
    Furan,
    Thiophene,
    Imidazole,
    Pyran,
    Cyclohexene,
    Custom { atoms: Vec<Element>, aromatic: bool },
}

/// Substituent type for attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubstituentType {
    Hydroxyl,
    Methyl,
    Ethyl,
    Propyl,
    Isopropyl,
    Butyl,
    Pentyl,
    Amino,
    Carboxyl,
    Nitro,
    Halogen(Element),
    Custom { smiles: String },
}

/// MOF size specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MOFSize {
    pub unit_cells_x: usize,
    pub unit_cells_y: usize,
    pub unit_cells_z: usize,
}

impl MOFSize {
    pub fn cube(size: usize) -> Self {
        Self {
            unit_cells_x: size,
            unit_cells_y: size,
            unit_cells_z: size,
        }
    }
}

/// COF dimension
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum COFDimension {
    TwoD,
    ThreeD,
}

/// Verification condition for conditional execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationCondition {
    StructureValid { molecule_id: String, tolerance: f64 },
    BondExists { atom1: AtomId, atom2: AtomId },
    AtomInPosition { atom_id: AtomId, position: Vector3<f64>, tolerance: f64 },
    StereocenterCorrect { atom_id: AtomId, config: StereoConfig },
    EnergyBelow { threshold_ev: f64 },
}

/// Molecular program (sequence of instructions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularProgram {
    pub name: String,
    pub instructions: Vec<MolecularInstruction>,
    pub next_instruction_id: u64,
}

impl MolecularProgram {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            instructions: Vec::new(),
            next_instruction_id: 1,
        }
    }

    /// Add instruction and return its ID
    pub fn add(&mut self, instruction: MolecularInstruction) -> u64 {
        let id = self.next_instruction_id;
        self.next_instruction_id += 1;

        // Update instruction ID
        let instruction = self.with_id(instruction, id);
        self.instructions.push(instruction);
        id
    }

    fn with_id(&self, mut instruction: MolecularInstruction, id: u64) -> MolecularInstruction {
        match &mut instruction {
            MolecularInstruction::PlaceAtom { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::FormBond { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::BreakBond { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::MoveAtom { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::RotateMolecule { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::LaserPulse { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::VerifyStructure { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::VerifyStereocenter { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::AssembleRing { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::BuildScaffold { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::AttachSubstituent { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::SynthesizeDNA { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::AssistProteinFolding { instruction_id, .. } => {
                *instruction_id = id
            }
            MolecularInstruction::BuildMOF { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::BuildCOF { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::BuildZIF { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::Wait { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::Checkpoint { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::Parallel { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::Conditional { instruction_id, .. } => *instruction_id = id,
            MolecularInstruction::Loop { instruction_id, .. } => *instruction_id = id,
        }
        instruction
    }

    /// Place atom instruction
    pub fn place_atom(&mut self, element: Element, position: Vector3<f64>) -> u64 {
        self.add(MolecularInstruction::PlaceAtom {
            instruction_id: 0,
            element,
            position,
            robot_type: RobotType::NanoQuantumonas,
            quantum_precision: true,
        })
    }

    /// Form bond instruction
    pub fn form_bond(&mut self, atom1: AtomId, atom2: AtomId, bond_type: BondType) -> u64 {
        self.add(MolecularInstruction::FormBond {
            instruction_id: 0,
            atom1_id: atom1,
            atom2_id: atom2,
            bond_type,
            use_laser: bond_type == BondType::Double || bond_type == BondType::Triple,
        })
    }

    /// Verify structure instruction
    pub fn verify_structure(&mut self, molecule_id: &str, atoms: usize, bonds: usize) -> u64 {
        self.add(MolecularInstruction::VerifyStructure {
            instruction_id: 0,
            molecule_id: molecule_id.to_string(),
            expected_atoms: atoms,
            expected_bonds: bonds,
            tolerance: 0.001,
        })
    }

    /// Checkpoint instruction
    pub fn checkpoint(&mut self, description: &str) -> u64 {
        self.add(MolecularInstruction::Checkpoint {
            instruction_id: 0,
            description: description.to_string(),
        })
    }

    /// Optimize instruction order for minimal robot movement
    pub fn optimize(&mut self) {
        // Group instructions by position to minimize movement
        // This is a simplified optimization - real implementation would use
        // more sophisticated algorithms

        let mut atom_placements: Vec<_> = self
            .instructions
            .iter()
            .filter(|i| matches!(i, MolecularInstruction::PlaceAtom { .. }))
            .cloned()
            .collect();

        // Sort by position to minimize travel
        atom_placements.sort_by(|a, b| {
            if let (
                MolecularInstruction::PlaceAtom { position: pa, .. },
                MolecularInstruction::PlaceAtom { position: pb, .. },
            ) = (a, b)
            {
                pa.x.partial_cmp(&pb.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(pa.y.partial_cmp(&pb.y).unwrap_or(std::cmp::Ordering::Equal))
                    .then(pa.z.partial_cmp(&pb.z).unwrap_or(std::cmp::Ordering::Equal))
            } else {
                std::cmp::Ordering::Equal
            }
        });

        // Collect non-placement instructions
        let other_instructions: Vec<_> = self
            .instructions
            .iter()
            .filter(|i| !matches!(i, MolecularInstruction::PlaceAtom { .. }))
            .cloned()
            .collect();

        // Rebuild instruction list
        self.instructions = atom_placements;
        self.instructions.extend(other_instructions);
    }

    /// Estimate execution time in milliseconds
    pub fn estimate_time(&self) -> u64 {
        let mut total_ms = 0u64;

        for instruction in &self.instructions {
            total_ms += match instruction {
                MolecularInstruction::PlaceAtom { .. } => 10,
                MolecularInstruction::FormBond { .. } => 5,
                MolecularInstruction::BreakBond { .. } => 8,
                MolecularInstruction::MoveAtom { .. } => 3,
                MolecularInstruction::RotateMolecule { .. } => 15,
                MolecularInstruction::LaserPulse { .. } => 1,
                MolecularInstruction::VerifyStructure { .. } => 50,
                MolecularInstruction::VerifyStereocenter { .. } => 20,
                MolecularInstruction::AssembleRing { .. } => 100,
                MolecularInstruction::BuildScaffold { .. } => 500,
                MolecularInstruction::AttachSubstituent { .. } => 30,
                MolecularInstruction::SynthesizeDNA { sequence, .. } => sequence.len() as u64 * 2,
                MolecularInstruction::AssistProteinFolding { sequence, .. } => {
                    sequence.len() as u64 * 10
                }
                MolecularInstruction::BuildMOF { size, .. } => {
                    (size.unit_cells_x * size.unit_cells_y * size.unit_cells_z) as u64 * 100
                }
                MolecularInstruction::BuildCOF { size, .. } => (*size * 50.0) as u64,
                MolecularInstruction::BuildZIF { .. } => 200,
                MolecularInstruction::Wait { duration_ms, .. } => *duration_ms,
                MolecularInstruction::Checkpoint { .. } => 5,
                MolecularInstruction::Parallel { instructions, .. } => {
                    instructions
                        .iter()
                        .map(|i| Self::instruction_time(i))
                        .max()
                        .unwrap_or(0)
                }
                MolecularInstruction::Conditional {
                    if_true, if_false, ..
                } => {
                    let true_time: u64 = if_true.iter().map(Self::instruction_time).sum();
                    let false_time: u64 = if_false.iter().map(Self::instruction_time).sum();
                    true_time.max(false_time) + 10
                }
                MolecularInstruction::Loop {
                    count,
                    instructions,
                    ..
                } => {
                    let loop_time: u64 = instructions.iter().map(Self::instruction_time).sum();
                    loop_time * (*count as u64)
                }
            };
        }

        total_ms
    }

    fn instruction_time(instruction: &MolecularInstruction) -> u64 {
        match instruction {
            MolecularInstruction::PlaceAtom { .. } => 10,
            MolecularInstruction::FormBond { .. } => 5,
            _ => 20, // Default estimate
        }
    }

    /// Get required robot types
    pub fn required_robots(&self) -> Vec<RobotType> {
        let mut robots = std::collections::HashSet::new();

        for instruction in &self.instructions {
            robots.insert(RobotType::best_for_operation(instruction));
        }

        robots.into_iter().collect()
    }
}

/// Builder for molecular programs
pub struct MolecularProgramBuilder {
    program: MolecularProgram,
    current_atom_id: AtomId,
}

impl MolecularProgramBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            program: MolecularProgram::new(name),
            current_atom_id: 0,
        }
    }

    pub fn place_carbon(&mut self, x: f64, y: f64, z: f64) -> AtomId {
        let id = self.current_atom_id;
        self.current_atom_id += 1;
        self.program
            .place_atom(Element::Carbon, Vector3::new(x, y, z));
        id
    }

    pub fn place_hydrogen(&mut self, x: f64, y: f64, z: f64) -> AtomId {
        let id = self.current_atom_id;
        self.current_atom_id += 1;
        self.program
            .place_atom(Element::Hydrogen, Vector3::new(x, y, z));
        id
    }

    pub fn place_oxygen(&mut self, x: f64, y: f64, z: f64) -> AtomId {
        let id = self.current_atom_id;
        self.current_atom_id += 1;
        self.program
            .place_atom(Element::Oxygen, Vector3::new(x, y, z));
        id
    }

    pub fn place_nitrogen(&mut self, x: f64, y: f64, z: f64) -> AtomId {
        let id = self.current_atom_id;
        self.current_atom_id += 1;
        self.program
            .place_atom(Element::Nitrogen, Vector3::new(x, y, z));
        id
    }

    pub fn single_bond(&mut self, atom1: AtomId, atom2: AtomId) -> &mut Self {
        self.program.form_bond(atom1, atom2, BondType::Single);
        self
    }

    pub fn double_bond(&mut self, atom1: AtomId, atom2: AtomId) -> &mut Self {
        self.program.form_bond(atom1, atom2, BondType::Double);
        self
    }

    pub fn aromatic_bond(&mut self, atom1: AtomId, atom2: AtomId) -> &mut Self {
        self.program.form_bond(atom1, atom2, BondType::Aromatic);
        self
    }

    pub fn checkpoint(&mut self, description: &str) -> &mut Self {
        self.program.checkpoint(description);
        self
    }

    pub fn build(mut self) -> MolecularProgram {
        self.program.optimize();
        self.program
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_methane() {
        let mut builder = MolecularProgramBuilder::new("Methane");

        let c = builder.place_carbon(0.0, 0.0, 0.0);
        let h1 = builder.place_hydrogen(1.09, 0.0, 0.0);
        let h2 = builder.place_hydrogen(-0.36, 1.03, 0.0);
        let h3 = builder.place_hydrogen(-0.36, -0.51, 0.89);
        let h4 = builder.place_hydrogen(-0.36, -0.51, -0.89);

        builder
            .single_bond(c, h1)
            .single_bond(c, h2)
            .single_bond(c, h3)
            .single_bond(c, h4)
            .checkpoint("Methane complete");

        let program = builder.build();

        // 5 atoms + 4 bonds + 1 checkpoint = 10 instructions
        assert_eq!(program.instructions.len(), 10);
        assert!(program.estimate_time() > 0);
    }

    #[test]
    fn test_parallel_execution() {
        let mut program = MolecularProgram::new("Parallel Test");

        program.add(MolecularInstruction::Parallel {
            instruction_id: 0,
            instructions: vec![
                MolecularInstruction::PlaceAtom {
                    instruction_id: 0,
                    element: Element::Carbon,
                    position: Vector3::zeros(),
                    robot_type: RobotType::NanoQuantumonas,
                    quantum_precision: true,
                },
                MolecularInstruction::PlaceAtom {
                    instruction_id: 0,
                    element: Element::Carbon,
                    position: Vector3::new(1.54, 0.0, 0.0),
                    robot_type: RobotType::NanoQuantumonas,
                    quantum_precision: true,
                },
            ],
        });

        assert_eq!(program.instructions.len(), 1);
    }
}
