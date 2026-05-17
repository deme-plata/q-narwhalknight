//! BioDSL Compiler
//!
//! Compiles BioDSL AST into Molecular IR for robot execution.

use crate::ast::*;
use crate::molecular_ir::*;
use crate::types::*;
use crate::{BioDSLError, CompiledProgram, SafetyConstraint, SafetyConstraintType};
use nalgebra::Vector3;

/// BioDSL compiler
pub struct BioDSLCompiler {
    next_atom_id: AtomId,
}

impl BioDSLCompiler {
    pub fn new() -> Self {
        Self { next_atom_id: 0 }
    }

    /// Compile a BioDSL program to molecular instructions
    pub fn compile(&self, program: BioDSLProgram) -> Result<CompiledProgram, BioDSLError> {
        let mut instructions = Vec::new();
        let mut safety_constraints = Vec::new();
        let mut required_robots = Vec::new();

        // Compile definitions
        for def in &program.definitions {
            match def {
                Definition::Molecule(mol) => {
                    let (mol_instructions, mol_safety) = self.compile_molecule(mol)?;
                    instructions.extend(mol_instructions);
                    safety_constraints.extend(mol_safety);
                    required_robots.push("NanoQuantumonas".to_string());
                }
                Definition::GeneticCircuit(circuit) => {
                    let (circuit_instructions, circuit_safety) =
                        self.compile_genetic_circuit(circuit)?;
                    instructions.extend(circuit_instructions);
                    safety_constraints.extend(circuit_safety);
                    required_robots.push("NanoQuantumonas".to_string());
                }
                Definition::Protein(protein) => {
                    let (protein_instructions, _) = self.compile_protein(protein)?;
                    instructions.extend(protein_instructions);
                    required_robots.push("CyberCetus".to_string());
                }
                Definition::Library(_library) => {
                    // Libraries generate multiple molecule compilations
                    // TODO: Implement library compilation
                }
            }
        }

        // Compile commands
        for cmd in &program.commands {
            match cmd {
                Command::Synthesize(synth) => {
                    let synth_instructions = self.compile_synthesize(synth, &program)?;
                    instructions.extend(synth_instructions);
                }
                Command::CompileToDNA(dna_cmd) => {
                    let dna_instructions = self.compile_to_dna(dna_cmd, &program)?;
                    instructions.extend(dna_instructions);
                }
                Command::Transform(transform) => {
                    let transform_instructions = self.compile_transform(transform)?;
                    instructions.extend(transform_instructions);
                }
                Command::Verify(verify) => {
                    let verify_instructions = self.compile_verify(verify)?;
                    instructions.extend(verify_instructions);
                }
                Command::Print(_print) => {
                    // Bio-printing is a separate subsystem
                    // TODO: Implement bio-printing compilation
                }
            }
        }

        // Estimate execution time
        let estimated_time_ms = self.estimate_time(&instructions);

        // Deduplicate required robots
        required_robots.sort();
        required_robots.dedup();

        Ok(CompiledProgram {
            instructions,
            safety_constraints,
            estimated_time_ms,
            required_robots,
        })
    }

    /// Compile molecule definition
    fn compile_molecule(
        &self,
        mol: &MoleculeDefinition,
    ) -> Result<(Vec<MolecularInstruction>, Vec<SafetyConstraint>), BioDSLError> {
        let mut instructions = Vec::new();
        let mut instruction_id = 1u64;
        let safety_constraints = Vec::new();

        // If SMILES is provided, parse and generate instructions
        if let Some(smiles) = &mol.smiles {
            let (smiles_instructions, _) = self.compile_smiles(smiles, &mut instruction_id)?;
            instructions.extend(smiles_instructions);
        }

        // If scaffold is provided, build from scaffold
        if let Some(scaffold) = &mol.scaffold {
            let scaffold_instructions = self.compile_scaffold(scaffold, &mut instruction_id)?;
            instructions.extend(scaffold_instructions);
        }

        // Add substituents
        for sub in &mol.substituents {
            let sub_instructions = self.compile_substituent(sub, &mut instruction_id)?;
            instructions.extend(sub_instructions);
        }

        // Verify stereocenters
        for stereo in &mol.stereocenters {
            instructions.push(MolecularInstruction::VerifyStereocenter {
                instruction_id,
                center_atom: self.parse_stereocenter_atom(&stereo.name)?,
                expected_config: stereo.config.clone(),
            });
            instruction_id += 1;
        }

        // Add verification step
        if let Some(verification) = &mol.verification {
            instructions.push(MolecularInstruction::VerifyStructure {
                instruction_id,
                molecule_id: mol.name.clone(),
                expected_atoms: 0, // Will be filled in
                expected_bonds: 0,
                tolerance: verification.tolerance,
            });
        }

        Ok((instructions, safety_constraints))
    }

    /// Compile SMILES notation to instructions
    fn compile_smiles(
        &self,
        smiles: &str,
        instruction_id: &mut u64,
    ) -> Result<(Vec<MolecularInstruction>, Vec<AtomId>), BioDSLError> {
        let mut instructions = Vec::new();
        let mut atom_ids = Vec::new();
        let mut atom_stack: Vec<(AtomId, Vector3<f64>)> = Vec::new();
        let mut current_position = Vector3::zeros();
        let mut current_atom_id = 0u32;
        let mut ring_atoms: std::collections::HashMap<char, AtomId> = std::collections::HashMap::new();

        let chars: Vec<char> = smiles.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            match c {
                'C' => {
                    // Carbon (check for Cl)
                    if i + 1 < chars.len() && chars[i + 1] == 'l' {
                        instructions.push(MolecularInstruction::PlaceAtom {
                            instruction_id: *instruction_id,
                            element: Element::Chlorine,
                            position: current_position,
                            robot_type: RobotType::NanoQuantumonas,
                            quantum_precision: true,
                        });
                        i += 1;
                    } else {
                        instructions.push(MolecularInstruction::PlaceAtom {
                            instruction_id: *instruction_id,
                            element: Element::Carbon,
                            position: current_position,
                            robot_type: RobotType::NanoQuantumonas,
                            quantum_precision: true,
                        });
                    }
                    *instruction_id += 1;

                    // Form bond to previous atom if any
                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_stack.push((current_atom_id, current_position));
                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                    current_position.x += 1.54; // C-C bond length
                }
                'N' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Nitrogen,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_stack.push((current_atom_id, current_position));
                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                    current_position.x += 1.47;
                }
                'O' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Oxygen,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_stack.push((current_atom_id, current_position));
                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                    current_position.x += 1.43;
                }
                'S' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Sulfur,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_stack.push((current_atom_id, current_position));
                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                    current_position.x += 1.81;
                }
                'F' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Fluorine,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                }
                'B' if i + 1 < chars.len() && chars[i + 1] == 'r' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Bromine,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;
                    i += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                }
                'I' => {
                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element: Element::Iodine,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Single,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                }
                '(' => {
                    // Branch start - save current position
                    // Position already on stack from last atom
                }
                ')' => {
                    // Branch end - pop back to branch point
                    if atom_stack.len() > 1 {
                        atom_stack.pop();
                        if let Some((_, pos)) = atom_stack.last() {
                            current_position = *pos;
                            current_position.y += 1.54; // Branch in Y direction
                        }
                    }
                }
                '=' => {
                    // Double bond - modify the last bond
                    if let Some(MolecularInstruction::FormBond { bond_type, .. }) =
                        instructions.last_mut()
                    {
                        *bond_type = BondType::Double;
                    }
                }
                '#' => {
                    // Triple bond
                    if let Some(MolecularInstruction::FormBond { bond_type, .. }) =
                        instructions.last_mut()
                    {
                        *bond_type = BondType::Triple;
                    }
                }
                '1'..='9' => {
                    // Ring closure
                    if let Some(&ring_start) = ring_atoms.get(&c) {
                        // Close ring
                        if !atom_stack.is_empty() {
                            let (current_id, _) = atom_stack.last().unwrap();
                            instructions.push(MolecularInstruction::FormBond {
                                instruction_id: *instruction_id,
                                atom1_id: ring_start,
                                atom2_id: *current_id,
                                bond_type: BondType::Single,
                                use_laser: false,
                            });
                            *instruction_id += 1;
                        }
                        ring_atoms.remove(&c);
                    } else if !atom_stack.is_empty() {
                        // Mark ring start
                        let (current_id, _) = atom_stack.last().unwrap();
                        ring_atoms.insert(c, *current_id);
                    }
                }
                'c' | 'n' | 'o' | 's' => {
                    // Aromatic atoms
                    let element = match c {
                        'c' => Element::Carbon,
                        'n' => Element::Nitrogen,
                        'o' => Element::Oxygen,
                        's' => Element::Sulfur,
                        _ => Element::Carbon,
                    };

                    instructions.push(MolecularInstruction::PlaceAtom {
                        instruction_id: *instruction_id,
                        element,
                        position: current_position,
                        robot_type: RobotType::NanoQuantumonas,
                        quantum_precision: true,
                    });
                    *instruction_id += 1;

                    if !atom_stack.is_empty() {
                        let (prev_id, _) = atom_stack.last().unwrap();
                        instructions.push(MolecularInstruction::FormBond {
                            instruction_id: *instruction_id,
                            atom1_id: *prev_id,
                            atom2_id: current_atom_id,
                            bond_type: BondType::Aromatic,
                            use_laser: false,
                        });
                        *instruction_id += 1;
                    }

                    atom_stack.push((current_atom_id, current_position));
                    atom_ids.push(current_atom_id);
                    current_atom_id += 1;
                    current_position.x += 1.40; // Aromatic bond length
                }
                _ => {
                    // Skip other characters (H, brackets, charges, etc.)
                }
            }

            i += 1;
        }

        Ok((instructions, atom_ids))
    }

    /// Compile scaffold definition
    fn compile_scaffold(
        &self,
        scaffold: &ScaffoldDefinition,
        instruction_id: &mut u64,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        let mut instructions = Vec::new();

        instructions.push(MolecularInstruction::BuildScaffold {
            instruction_id: *instruction_id,
            scaffold_name: scaffold.name.clone(),
            position: Vector3::zeros(),
            scale: 1.0,
        });
        *instruction_id += 1;

        // Build individual rings
        for ring in &scaffold.rings {
            let ring_type = match &ring.ring_type {
                RingTypeSpec::Benzene => RingAssemblyType::Benzene,
                RingTypeSpec::Pyran => RingAssemblyType::Pyran,
                RingTypeSpec::Cyclohexene => RingAssemblyType::Cyclohexene,
                RingTypeSpec::Cyclohexane => RingAssemblyType::Cyclohexane,
                RingTypeSpec::Cyclopentane => RingAssemblyType::Cyclopentane,
                RingTypeSpec::Pyridine => RingAssemblyType::Pyridine,
                RingTypeSpec::Pyrrole => RingAssemblyType::Pyrrole,
                RingTypeSpec::Furan => RingAssemblyType::Furan,
                RingTypeSpec::Thiophene => RingAssemblyType::Thiophene,
                RingTypeSpec::Imidazole => RingAssemblyType::Imidazole,
                RingTypeSpec::Custom { size, aromatic } => RingAssemblyType::Custom {
                    atoms: vec![Element::Carbon; *size],
                    aromatic: *aromatic,
                },
            };

            let position = ring
                .position
                .as_ref()
                .map(|p| Vector3::new(p.x, p.y, p.z))
                .unwrap_or_else(Vector3::zeros);

            instructions.push(MolecularInstruction::AssembleRing {
                instruction_id: *instruction_id,
                ring_type,
                center: position,
                orientation: Vector3::new(0.0, 0.0, 1.0),
            });
            *instruction_id += 1;
        }

        Ok(instructions)
    }

    /// Compile substituent
    fn compile_substituent(
        &self,
        sub: &SubstituentDefinition,
        instruction_id: &mut u64,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        let mut instructions = Vec::new();

        let substituent_type = match &sub.group {
            FunctionalGroup::Hydroxyl => SubstituentType::Hydroxyl,
            FunctionalGroup::Methyl => SubstituentType::Methyl,
            FunctionalGroup::Ethyl => SubstituentType::Ethyl,
            FunctionalGroup::Propyl => SubstituentType::Propyl,
            FunctionalGroup::Isopropyl => SubstituentType::Isopropyl,
            FunctionalGroup::Butyl => SubstituentType::Butyl,
            FunctionalGroup::Pentyl => SubstituentType::Pentyl,
            FunctionalGroup::Amino => SubstituentType::Amino,
            FunctionalGroup::Carboxyl => SubstituentType::Carboxyl,
            FunctionalGroup::Nitro => SubstituentType::Nitro,
            FunctionalGroup::Halogen(elem) => SubstituentType::Halogen(*elem),
            FunctionalGroup::Custom(name) => SubstituentType::Custom {
                smiles: name.clone(),
            },
            _ => SubstituentType::Methyl,
        };

        // Get target atom(s)
        let target_atoms = match &sub.position {
            SubstituentPosition::AtomId(id) => vec![*id],
            SubstituentPosition::RingPosition { position, .. } => vec![*position as AtomId],
            SubstituentPosition::Named(_) => vec![0], // Will be resolved later
            SubstituentPosition::Multiple(positions) => positions
                .iter()
                .filter_map(|p| match p {
                    SubstituentPosition::AtomId(id) => Some(*id),
                    SubstituentPosition::RingPosition { position, .. } => Some(*position as AtomId),
                    _ => None,
                })
                .collect(),
        };

        for target_atom in target_atoms {
            for _ in 0..sub.count {
                instructions.push(MolecularInstruction::AttachSubstituent {
                    instruction_id: *instruction_id,
                    target_atom,
                    substituent: substituent_type.clone(),
                    orientation: Vector3::new(1.0, 0.0, 0.0),
                });
                *instruction_id += 1;
            }
        }

        Ok(instructions)
    }

    /// Compile genetic circuit
    fn compile_genetic_circuit(
        &self,
        circuit: &GeneticCircuitDefinition,
    ) -> Result<(Vec<MolecularInstruction>, Vec<SafetyConstraint>), BioDSLError> {
        let mut instructions = Vec::new();
        let mut safety_constraints = Vec::new();
        let mut instruction_id = 1u64;

        // Build DNA sequence for each gene
        for gene in &circuit.genes {
            // Synthesize promoter
            if let Some(promoter) = circuit.promoters.iter().find(|p| p.name == gene.promoter) {
                if let Some(seq) = &promoter.sequence {
                    instructions.push(MolecularInstruction::SynthesizeDNA {
                        instruction_id,
                        sequence: seq.clone(),
                        start_position: Vector3::zeros(),
                    });
                    instruction_id += 1;
                }
            }

            // Synthesize gene sequence
            if let Some(seq) = &gene.sequence {
                instructions.push(MolecularInstruction::SynthesizeDNA {
                    instruction_id,
                    sequence: seq.clone(),
                    start_position: Vector3::zeros(),
                });
                instruction_id += 1;
            }
        }

        // Add safety constraints
        if let Some(safety) = &circuit.safety {
            for aux in &safety.auxotrophy {
                safety_constraints.push(SafetyConstraint {
                    constraint_type: SafetyConstraintType::EnvironmentalContainment,
                    molecule_id: Some(aux.clone()),
                    max_quantity: None,
                });
            }

            if !safety.kill_switches.is_empty() {
                safety_constraints.push(SafetyConstraint {
                    constraint_type: SafetyConstraintType::KillSwitchRequired,
                    molecule_id: Some(circuit.name.clone()),
                    max_quantity: None,
                });
            }

            if let Some(gen_limit) = safety.generation_limit {
                safety_constraints.push(SafetyConstraint {
                    constraint_type: SafetyConstraintType::QuantityLimit,
                    molecule_id: Some(circuit.name.clone()),
                    max_quantity: Some(gen_limit as f64),
                });
            }
        }

        Ok((instructions, safety_constraints))
    }

    /// Compile protein definition
    fn compile_protein(
        &self,
        protein: &ProteinDefinition,
    ) -> Result<(Vec<MolecularInstruction>, Vec<SafetyConstraint>), BioDSLError> {
        let mut instructions = Vec::new();

        instructions.push(MolecularInstruction::AssistProteinFolding {
            instruction_id: 1,
            sequence: protein.sequence.clone(),
            target_structure_id: None,
        });

        Ok((instructions, Vec::new()))
    }

    /// Compile synthesize command
    fn compile_synthesize(
        &self,
        cmd: &SynthesizeCommand,
        _program: &BioDSLProgram,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        let mut instructions = Vec::new();

        // Add checkpoint for synthesis start
        instructions.push(MolecularInstruction::Checkpoint {
            instruction_id: 1,
            description: format!("Begin synthesis of {}", cmd.target),
        });

        // Add verification at end
        instructions.push(MolecularInstruction::VerifyStructure {
            instruction_id: 2,
            molecule_id: cmd.target.clone(),
            expected_atoms: 0,
            expected_bonds: 0,
            tolerance: cmd.purity.unwrap_or(0.99),
        });

        Ok(instructions)
    }

    /// Compile to DNA command
    fn compile_to_dna(
        &self,
        cmd: &CompileToDNACommand,
        _program: &BioDSLProgram,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        Ok(vec![MolecularInstruction::Checkpoint {
            instruction_id: 1,
            description: format!("Compile {} to DNA", cmd.circuit),
        }])
    }

    /// Compile transform command
    fn compile_transform(
        &self,
        cmd: &TransformCommand,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        Ok(vec![MolecularInstruction::Checkpoint {
            instruction_id: 1,
            description: format!("Transform {} with {} using {:?}", cmd.host, cmd.plasmid, cmd.method),
        }])
    }

    /// Compile verify command
    fn compile_verify(
        &self,
        cmd: &VerifyCommand,
    ) -> Result<Vec<MolecularInstruction>, BioDSLError> {
        Ok(vec![MolecularInstruction::VerifyStructure {
            instruction_id: 1,
            molecule_id: cmd.target.clone(),
            expected_atoms: 0,
            expected_bonds: 0,
            tolerance: cmd.tolerance,
        }])
    }

    /// Parse stereocenter atom name to ID
    fn parse_stereocenter_atom(&self, name: &str) -> Result<AtomId, BioDSLError> {
        // Try to parse as number
        if let Ok(id) = name.parse::<u32>() {
            return Ok(id);
        }

        // Handle named stereocenters like "6a", "10a"
        let numeric_part: String = name.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(id) = numeric_part.parse::<u32>() {
            return Ok(id);
        }

        Err(BioDSLError::InvalidStereochemistry(format!(
            "Cannot parse stereocenter: {}",
            name
        )))
    }

    /// Estimate execution time
    fn estimate_time(&self, instructions: &[MolecularInstruction]) -> u64 {
        let mut total = 0u64;
        for instr in instructions {
            total += match instr {
                MolecularInstruction::PlaceAtom { .. } => 10,
                MolecularInstruction::FormBond { .. } => 5,
                MolecularInstruction::VerifyStructure { .. } => 50,
                MolecularInstruction::SynthesizeDNA { sequence, .. } => sequence.len() as u64 * 2,
                MolecularInstruction::AssistProteinFolding { sequence, .. } => {
                    sequence.len() as u64 * 10
                }
                _ => 20,
            };
        }
        total
    }
}

impl Default for BioDSLCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BioDSL;

    #[test]
    fn test_compile_simple_molecule() {
        let source = r#"
            molecule Water {
                smiles: "O"
            }
        "#;

        let program = BioDSL::parse(source).unwrap();
        let compiler = BioDSLCompiler::new();
        let compiled = compiler.compile(program).unwrap();

        assert!(!compiled.instructions.is_empty());
    }

    #[test]
    fn test_compile_smiles() {
        let compiler = BioDSLCompiler::new();
        let mut instruction_id = 1;

        let (instructions, _) = compiler.compile_smiles("CCO", &mut instruction_id).unwrap();

        // Should have atom placements and bond formations
        assert!(!instructions.is_empty());
    }
}
