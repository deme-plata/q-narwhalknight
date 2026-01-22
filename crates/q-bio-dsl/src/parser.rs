//! Parser for BioDSL
//!
//! Parses tokenized BioDSL source into an AST.

use crate::ast::*;
use crate::lexer::{BioDSLLexer, SpannedToken, Token};
use crate::types::*;
use crate::BioDSLError;

/// BioDSL parser
pub struct BioDSLParser {
    tokens: Vec<SpannedToken>,
    position: usize,
}

impl BioDSLParser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            position: 0,
        }
    }

    /// Parse source code into a program
    pub fn parse(&self, source: &str) -> Result<BioDSLProgram, BioDSLError> {
        let tokens =
            BioDSLLexer::tokenize(source).map_err(|e| BioDSLError::ParseError(e.to_string()))?;

        let mut parser = Self {
            tokens,
            position: 0,
        };

        parser.parse_program()
    }

    fn parse_program(&mut self) -> Result<BioDSLProgram, BioDSLError> {
        let mut program = BioDSLProgram::new();

        while !self.is_at_end() {
            if self.check(&Token::Molecule) {
                let mol = self.parse_molecule_definition()?;
                program.add_definition(Definition::Molecule(mol));
            } else if self.check(&Token::GeneticCircuit) {
                let circuit = self.parse_genetic_circuit()?;
                program.add_definition(Definition::GeneticCircuit(circuit));
            } else if self.check(&Token::Protein) {
                let protein = self.parse_protein_definition()?;
                program.add_definition(Definition::Protein(protein));
            } else if self.check(&Token::Synthesize) || self.check(&Token::RobotSwarm) {
                let cmd = self.parse_command()?;
                program.add_command(cmd);
            } else if let Some(Token::Identifier(_)) = self.peek_token() {
                // Could be a command or expression
                let cmd = self.parse_command()?;
                program.add_command(cmd);
            } else {
                self.advance();
            }
        }

        Ok(program)
    }

    /// Parse molecule definition
    fn parse_molecule_definition(&mut self) -> Result<MoleculeDefinition, BioDSLError> {
        self.expect(&Token::Molecule)?;

        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut mol = MoleculeDefinition::new(&name);

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Smiles) {
                self.advance();
                self.expect(&Token::Colon)?;
                let smiles = self.expect_string()?;
                mol.smiles = Some(smiles);
            } else if self.check(&Token::Scaffold) {
                mol.scaffold = Some(self.parse_scaffold()?);
            } else if self.check(&Token::Substituents) {
                self.advance();
                self.expect(&Token::LBrace)?;
                while !self.check(&Token::RBrace) && !self.is_at_end() {
                    let sub = self.parse_substituent()?;
                    mol.substituents.push(sub);
                }
                self.expect(&Token::RBrace)?;
            } else if self.check(&Token::Stereochemistry) {
                self.advance();
                self.expect(&Token::LBrace)?;
                while !self.check(&Token::RBrace) && !self.is_at_end() {
                    let stereo = self.parse_stereocenter()?;
                    mol.stereocenters.push(stereo);
                }
                self.expect(&Token::RBrace)?;
            } else if self.check(&Token::SynthesisMethod) {
                self.advance();
                self.expect(&Token::Colon)?;
                mol.synthesis_method = Some(self.parse_synthesis_method()?);
            } else if self.check(&Token::Verification) {
                self.advance();
                self.expect(&Token::Colon)?;
                mol.verification = Some(self.parse_verification()?);
            } else {
                // Skip unknown tokens within molecule block
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(mol)
    }

    /// Parse scaffold definition
    fn parse_scaffold(&mut self) -> Result<ScaffoldDefinition, BioDSLError> {
        self.expect(&Token::Scaffold)?;
        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut scaffold = ScaffoldDefinition {
            name,
            rings: Vec::new(),
        };

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Ring) {
                self.advance();
                let ring = self.parse_ring()?;
                scaffold.rings.push(ring);
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(scaffold)
    }

    /// Parse ring definition
    fn parse_ring(&mut self) -> Result<RingDefinition, BioDSLError> {
        let name = self.expect_identifier()?;
        self.expect(&Token::Colon)?;

        let ring_type = self.parse_ring_type()?;

        let mut ring = RingDefinition {
            name,
            ring_type,
            position: None,
            fused_to: None,
        };

        // Check for position or fusion
        if self.check(&Token::At) {
            self.advance();
            if self.check_identifier() {
                let attr = self.expect_identifier()?;
                if attr == "position" {
                    ring.position = Some(self.parse_position()?);
                } else if attr == "fused_to" {
                    ring.fused_to = Some(self.parse_fusion()?);
                }
            }
        }

        Ok(ring)
    }

    /// Parse ring type
    fn parse_ring_type(&mut self) -> Result<RingTypeSpec, BioDSLError> {
        if self.check(&Token::Benzene) {
            self.advance();
            Ok(RingTypeSpec::Benzene)
        } else if self.check(&Token::Pyran) {
            self.advance();
            Ok(RingTypeSpec::Pyran)
        } else if self.check(&Token::Cyclohexene) {
            self.advance();
            Ok(RingTypeSpec::Cyclohexene)
        } else {
            let name = self.expect_identifier()?;
            match name.to_lowercase().as_str() {
                "benzene" => Ok(RingTypeSpec::Benzene),
                "pyran" => Ok(RingTypeSpec::Pyran),
                "cyclohexene" => Ok(RingTypeSpec::Cyclohexene),
                "cyclohexane" => Ok(RingTypeSpec::Cyclohexane),
                "cyclopentane" => Ok(RingTypeSpec::Cyclopentane),
                "pyridine" => Ok(RingTypeSpec::Pyridine),
                "pyrrole" => Ok(RingTypeSpec::Pyrrole),
                "furan" => Ok(RingTypeSpec::Furan),
                "thiophene" => Ok(RingTypeSpec::Thiophene),
                "imidazole" => Ok(RingTypeSpec::Imidazole),
                _ => Err(BioDSLError::ParseError(format!(
                    "Unknown ring type: {}",
                    name
                ))),
            }
        }
    }

    /// Parse position specification
    fn parse_position(&mut self) -> Result<PositionSpec, BioDSLError> {
        self.expect(&Token::LParen)?;
        let x = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let y = self.expect_float()?;
        self.expect(&Token::Comma)?;
        let z = self.expect_float()?;
        self.expect(&Token::RParen)?;
        Ok(PositionSpec::new(x, y, z))
    }

    /// Parse fusion specification
    fn parse_fusion(&mut self) -> Result<FusionSpec, BioDSLError> {
        self.expect(&Token::LParen)?;
        let target_ring = self.expect_identifier()?;

        let mut positions = Vec::new();
        if self.check(&Token::Comma) {
            self.advance();
            // Parse positions: keyword
            self.expect_identifier()?; // "positions"
            self.expect(&Token::Colon)?;
            self.expect(&Token::LBracket)?;
            while !self.check(&Token::RBracket) {
                let pos = self.expect_int()? as usize;
                positions.push(pos);
                if self.check(&Token::Comma) {
                    self.advance();
                }
            }
            self.expect(&Token::RBracket)?;
        }

        self.expect(&Token::RParen)?;
        Ok(FusionSpec {
            target_ring,
            positions,
        })
    }

    /// Parse substituent definition
    fn parse_substituent(&mut self) -> Result<SubstituentDefinition, BioDSLError> {
        let group = self.parse_functional_group()?;

        let mut count = 1;
        if self.check(&Token::LBracket) {
            self.advance();
            count = self.expect_int()? as usize;
            self.expect(&Token::RBracket)?;
        }

        self.expect(&Token::At)?;
        let position = self.parse_substituent_position()?;

        Ok(SubstituentDefinition {
            group,
            position,
            count,
        })
    }

    /// Parse functional group
    fn parse_functional_group(&mut self) -> Result<FunctionalGroup, BioDSLError> {
        if self.check(&Token::Hydroxyl) {
            self.advance();
            Ok(FunctionalGroup::Hydroxyl)
        } else if self.check(&Token::Methyl) {
            self.advance();
            Ok(FunctionalGroup::Methyl)
        } else if self.check(&Token::Ethyl) {
            self.advance();
            Ok(FunctionalGroup::Ethyl)
        } else if self.check(&Token::Pentyl) {
            self.advance();
            Ok(FunctionalGroup::Pentyl)
        } else if self.check(&Token::Amino) {
            self.advance();
            Ok(FunctionalGroup::Amino)
        } else if self.check(&Token::Carboxyl) {
            self.advance();
            Ok(FunctionalGroup::Carboxyl)
        } else {
            let name = self.expect_identifier()?;
            match name.to_lowercase().as_str() {
                "hydroxyl" | "oh" => Ok(FunctionalGroup::Hydroxyl),
                "methyl" | "ch3" => Ok(FunctionalGroup::Methyl),
                "ethyl" | "c2h5" => Ok(FunctionalGroup::Ethyl),
                "propyl" | "c3h7" => Ok(FunctionalGroup::Propyl),
                "isopropyl" => Ok(FunctionalGroup::Isopropyl),
                "butyl" | "c4h9" => Ok(FunctionalGroup::Butyl),
                "pentyl" | "c5h11" => Ok(FunctionalGroup::Pentyl),
                "amino" | "nh2" => Ok(FunctionalGroup::Amino),
                "carboxyl" | "cooh" => Ok(FunctionalGroup::Carboxyl),
                "carbonyl" | "co" => Ok(FunctionalGroup::Carbonyl),
                "nitro" | "no2" => Ok(FunctionalGroup::Nitro),
                _ => Ok(FunctionalGroup::Custom(name)),
            }
        }
    }

    /// Parse substituent position
    fn parse_substituent_position(&mut self) -> Result<SubstituentPosition, BioDSLError> {
        let first = self.expect_identifier()?;

        if self.check(&Token::Dot) {
            self.advance();
            let attr = self.expect_identifier()?;
            if attr == "position" || attr == "positions" {
                self.expect(&Token::LParen)?;
                if self.check(&Token::LBracket) {
                    // Multiple positions
                    self.advance();
                    let mut positions = Vec::new();
                    while !self.check(&Token::RBracket) {
                        let pos = self.expect_int()? as usize;
                        positions.push(SubstituentPosition::RingPosition {
                            ring: first.clone(),
                            position: pos,
                        });
                        if self.check(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(&Token::RBracket)?;
                    self.expect(&Token::RParen)?;
                    return Ok(SubstituentPosition::Multiple(positions));
                } else {
                    let pos = self.expect_int()? as usize;
                    self.expect(&Token::RParen)?;
                    return Ok(SubstituentPosition::RingPosition {
                        ring: first,
                        position: pos,
                    });
                }
            }
        }

        Ok(SubstituentPosition::Named(first))
    }

    /// Parse stereocenter definition
    fn parse_stereocenter(&mut self) -> Result<StereocenterDefinition, BioDSLError> {
        if self.check(&Token::ChiralCenter) {
            self.advance();
        }

        let name = if self.check(&Token::LParen) {
            self.advance();
            let n = self.expect_identifier()?;
            self.expect(&Token::RParen)?;
            n
        } else {
            self.expect_identifier()?
        };

        self.expect(&Token::Colon)?;

        let config = if self.check(&Token::ConfigR) {
            self.advance();
            StereoConfig::R
        } else if self.check(&Token::ConfigS) {
            self.advance();
            StereoConfig::S
        } else if self.check(&Token::ConfigE) {
            self.advance();
            StereoConfig::E
        } else if self.check(&Token::ConfigZ) {
            self.advance();
            StereoConfig::Z
        } else {
            let config_str = self.expect_identifier()?;
            match config_str.to_uppercase().as_str() {
                "R" => StereoConfig::R,
                "S" => StereoConfig::S,
                "E" => StereoConfig::E,
                "Z" => StereoConfig::Z,
                _ => {
                    return Err(BioDSLError::InvalidStereochemistry(format!(
                        "Unknown config: {}",
                        config_str
                    )))
                }
            }
        };

        Ok(StereocenterDefinition { name, config })
    }

    /// Parse synthesis method specification
    fn parse_synthesis_method(&mut self) -> Result<SynthesisMethodSpec, BioDSLError> {
        let robot_type = self.expect_identifier()?;

        self.expect(&Token::Dot)?;
        let method = self.expect_identifier()?;

        Ok(SynthesisMethodSpec {
            robot_type,
            method,
            parameters: Vec::new(),
        })
    }

    /// Parse verification specification
    fn parse_verification(&mut self) -> Result<VerificationSpec, BioDSLError> {
        let method = self.expect_identifier()?;

        let mut tolerance = 0.001;
        if self.check(&Token::LParen) {
            self.advance();
            // Parse tolerance: value
            if self.check_identifier() {
                self.advance(); // "tolerance"
                self.expect(&Token::Colon)?;
                tolerance = self.expect_float()?;
            }
            self.expect(&Token::RParen)?;
        }

        Ok(VerificationSpec {
            method,
            tolerance,
            parameters: Vec::new(),
        })
    }

    /// Parse genetic circuit definition
    fn parse_genetic_circuit(&mut self) -> Result<GeneticCircuitDefinition, BioDSLError> {
        self.expect(&Token::GeneticCircuit)?;
        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut circuit = GeneticCircuitDefinition::new(&name);

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Promoter) {
                circuit.promoters.push(self.parse_promoter()?);
            } else if self.check(&Token::Gene) {
                circuit.genes.push(self.parse_gene()?);
            } else if self.check(&Token::Input) {
                circuit.inputs.push(self.parse_circuit_input()?);
            } else if self.check(&Token::Output) {
                circuit.outputs.push(self.parse_circuit_output()?);
            } else if self.check(&Token::Safety) {
                circuit.safety = Some(self.parse_safety_features()?);
            } else if let Some(Token::Identifier(_)) = self.peek_token() {
                // Could be interaction: lacI.represses(pTet)
                let interaction = self.parse_interaction()?;
                if let Some(i) = interaction {
                    circuit.interactions.push(i);
                }
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(circuit)
    }

    /// Parse promoter definition
    fn parse_promoter(&mut self) -> Result<PromoterDefinition, BioDSLError> {
        self.expect(&Token::Promoter)?;
        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut promoter = PromoterDefinition::new(&name);

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::BindingSite) {
                self.advance();
                self.expect(&Token::Colon)?;
                let site = self.expect_identifier()?;
                promoter.binding_sites.push(site);
            } else if self.check(&Token::Strength) {
                self.advance();
                self.expect(&Token::Colon)?;
                promoter.strength = self.expect_float()?;
            } else if self.check(&Token::Sequence) {
                self.advance();
                self.expect(&Token::Colon)?;
                promoter.sequence = Some(self.expect_string()?);
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(promoter)
    }

    /// Parse gene definition
    fn parse_gene(&mut self) -> Result<GeneDefinition, BioDSLError> {
        self.expect(&Token::Gene)?;
        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut promoter = String::new();
        let mut product = String::new();
        let mut degradation_time = None;

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Promoter) {
                self.advance();
                self.expect(&Token::Colon)?;
                promoter = self.expect_identifier()?;
            } else if self.check(&Token::Product) {
                self.advance();
                self.expect(&Token::Colon)?;
                product = self.expect_identifier()?;
            } else if self.check(&Token::DegradesIn) {
                self.advance();
                self.expect(&Token::Colon)?;
                let value = self.expect_float()?;
                let unit = if self.check(&Token::Minutes) {
                    self.advance();
                    TimeUnit::Minutes
                } else if self.check(&Token::Hours) {
                    self.advance();
                    TimeUnit::Hours
                } else if self.check(&Token::Seconds) {
                    self.advance();
                    TimeUnit::Seconds
                } else {
                    // Try to parse unit from identifier
                    let unit_str = self.expect_identifier()?;
                    match unit_str.to_lowercase().as_str() {
                        "minutes" | "min" => TimeUnit::Minutes,
                        "hours" | "hr" => TimeUnit::Hours,
                        "seconds" | "sec" => TimeUnit::Seconds,
                        _ => TimeUnit::Minutes,
                    }
                };
                degradation_time = Some(DurationSpec { value, unit });
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;

        let mut gene = GeneDefinition::new(&name, &promoter, &product);
        gene.degradation_time = degradation_time;
        Ok(gene)
    }

    /// Parse circuit input
    fn parse_circuit_input(&mut self) -> Result<CircuitInput, BioDSLError> {
        self.expect(&Token::Input)?;
        let name = self.expect_identifier()?;

        let action = if self.check(&Token::SwitchesOff) {
            self.advance();
            InputAction::SwitchesOff
        } else if self.check(&Token::SwitchesOn) {
            self.advance();
            InputAction::SwitchesOn
        } else {
            let action_str = self.expect_identifier()?;
            match action_str.as_str() {
                "switches_off" => InputAction::SwitchesOff,
                "switches_on" => InputAction::SwitchesOn,
                _ => InputAction::Modulates { factor: 1.0 },
            }
        };

        let target = self.expect_identifier()?;

        Ok(CircuitInput {
            name,
            action,
            target,
        })
    }

    /// Parse circuit output
    fn parse_circuit_output(&mut self) -> Result<CircuitOutput, BioDSLError> {
        self.expect(&Token::Output)?;
        let name = self.expect_identifier()?;

        let reporter_type = match name.to_uppercase().as_str() {
            "GFP" => ReporterType::GFP,
            "RFP" => ReporterType::RFP,
            "YFP" => ReporterType::YFP,
            "CFP" => ReporterType::CFP,
            "BFP" => ReporterType::BFP,
            "LUCIFERASE" => ReporterType::Luciferase,
            "LACZ" => ReporterType::LacZ,
            _ => ReporterType::Custom(name.clone()),
        };

        let mut fused_to = String::new();
        if self.check(&Token::FusedTo) {
            self.advance();
            fused_to = self.expect_identifier()?;
        } else if let Some(Token::Identifier(_)) = self.peek_token() {
            fused_to = self.expect_identifier()?;
        }

        Ok(CircuitOutput {
            name,
            reporter_type,
            fused_to,
        })
    }

    /// Parse safety features
    fn parse_safety_features(&mut self) -> Result<SafetyFeatures, BioDSLError> {
        self.expect(&Token::Safety)?;
        self.expect(&Token::LBrace)?;

        let mut safety = SafetyFeatures {
            auxotrophy: Vec::new(),
            kill_switches: Vec::new(),
            generation_limit: None,
            genetic_firewall: false,
        };

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Auxotrophy) {
                self.advance();
                self.expect(&Token::Colon)?;
                let aux = self.expect_identifier()?;
                safety.auxotrophy.push(aux);
            } else if self.check(&Token::KillSwitch) {
                self.advance();
                self.expect(&Token::Colon)?;
                let ks = self.parse_kill_switch()?;
                safety.kill_switches.push(ks);
            } else if self.check(&Token::GenerationLimit) {
                self.advance();
                self.expect(&Token::Colon)?;
                safety.generation_limit = Some(self.expect_int()? as u32);
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(safety)
    }

    /// Parse kill switch
    fn parse_kill_switch(&mut self) -> Result<KillSwitchSpec, BioDSLError> {
        let trigger_type = self.expect_identifier()?;

        let trigger = if trigger_type == "temperature_sensitive" {
            self.expect(&Token::LParen)?;
            // Parse max: value or min: value
            let mut min = f64::NEG_INFINITY;
            let mut max = f64::INFINITY;

            while !self.check(&Token::RParen) {
                let param = self.expect_identifier()?;
                self.expect(&Token::Colon)?;
                let value = self.expect_float()?;
                if param == "max" {
                    max = value;
                } else if param == "min" {
                    min = value;
                }
                if self.check(&Token::Comma) {
                    self.advance();
                }
            }
            self.expect(&Token::RParen)?;
            KillTriggerSpec::TemperatureSensitive { min, max }
        } else if trigger_type == "nutrient_dependent" {
            self.expect(&Token::LParen)?;
            let nutrient = self.expect_identifier()?;
            self.expect(&Token::RParen)?;
            KillTriggerSpec::NutrientDependent(nutrient)
        } else if trigger_type == "light_sensitive" {
            KillTriggerSpec::LightSensitive
        } else {
            KillTriggerSpec::ChemicalInducible(trigger_type)
        };

        Ok(KillSwitchSpec {
            trigger,
            toxin: "ccdB".to_string(), // Default toxin
            response_time: DurationSpec::minutes(30.0),
        })
    }

    /// Parse interaction (e.g., lacI.represses(pTet))
    fn parse_interaction(&mut self) -> Result<Option<GeneticInteraction>, BioDSLError> {
        let source = self.expect_identifier()?;

        if self.check(&Token::Dot) {
            self.advance();

            if self.check(&Token::Represses) {
                self.advance();
                self.expect(&Token::LParen)?;
                let target = self.expect_identifier()?;
                self.expect(&Token::RParen)?;
                return Ok(Some(GeneticInteraction::Repression {
                    repressor: source,
                    target,
                    strength: 1.0,
                }));
            } else if self.check(&Token::Activates) {
                self.advance();
                self.expect(&Token::LParen)?;
                let target = self.expect_identifier()?;
                self.expect(&Token::RParen)?;
                return Ok(Some(GeneticInteraction::Activation {
                    activator: source,
                    target,
                    strength: 1.0,
                }));
            }
        }

        Ok(None)
    }

    /// Parse protein definition
    fn parse_protein_definition(&mut self) -> Result<ProteinDefinition, BioDSLError> {
        self.expect(&Token::Protein)?;
        let name = self.expect_identifier()?;
        self.expect(&Token::LBrace)?;

        let mut sequence = String::new();
        let mut protein = ProteinDefinition::new(&name, "");

        while !self.check(&Token::RBrace) && !self.is_at_end() {
            if self.check(&Token::Sequence) {
                self.advance();
                self.expect(&Token::Colon)?;
                sequence = self.expect_string()?;
            } else {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;

        protein.sequence = sequence;
        Ok(protein)
    }

    /// Parse command
    fn parse_command(&mut self) -> Result<Command, BioDSLError> {
        if self.check(&Token::Synthesize) {
            self.advance();
            self.expect(&Token::LParen)?;
            let target = self.expect_identifier()?;

            let mut cmd = SynthesizeCommand::new(&target, Quantity::milligrams(1.0));

            while !self.check(&Token::RParen) && !self.is_at_end() {
                if self.check(&Token::Comma) {
                    self.advance();
                }
                if self.check(&Token::Amount) || self.check(&Token::QuantityKw) {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    let amount = self.expect_float()?;
                    let unit = self.parse_quantity_unit()?;
                    cmd.quantity = Quantity { amount, unit };
                } else if self.check(&Token::Purity) {
                    self.advance();
                    self.expect(&Token::Colon)?;
                    cmd.purity = Some(self.expect_float()?);
                } else if let Some(Token::Identifier(id)) = self.peek_token() {
                    if id == "amount" || id == "quantity" {
                        self.advance();
                        self.expect(&Token::Colon)?;
                        let amount = self.expect_float()?;
                        let unit = self.parse_quantity_unit()?;
                        cmd.quantity = Quantity { amount, unit };
                    } else if id == "purity" {
                        self.advance();
                        self.expect(&Token::Colon)?;
                        cmd.purity = Some(self.expect_float()?);
                    } else {
                        self.advance();
                    }
                } else {
                    self.advance();
                }
            }

            self.expect(&Token::RParen)?;
            Ok(Command::Synthesize(cmd))
        } else if self.check(&Token::RobotSwarm) {
            self.advance();
            self.expect(&Token::Dot)?;
            let method = self.expect_identifier()?;

            if method == "synthesize" {
                self.expect(&Token::LParen)?;
                let target = self.expect_identifier()?;

                let mut cmd = SynthesizeCommand::new(&target, Quantity::milligrams(1.0));

                while !self.check(&Token::RParen) && !self.is_at_end() {
                    if self.check(&Token::Comma) {
                        self.advance();
                    }
                    // Parse named parameters
                    if let Some(Token::Identifier(id)) = self.peek_token() {
                        if id == "quantity" || id == "amount" {
                            self.advance();
                            self.expect(&Token::Colon)?;
                            let amount = self.expect_float()?;
                            let unit = self.parse_quantity_unit()?;
                            cmd.quantity = Quantity { amount, unit };
                        } else if id == "purity" {
                            self.advance();
                            self.expect(&Token::Colon)?;
                            cmd.purity = Some(self.expect_float()?);
                        } else {
                            self.advance();
                        }
                    } else {
                        self.advance();
                    }
                }

                self.expect(&Token::RParen)?;
                return Ok(Command::Synthesize(cmd));
            }

            // Generic command fallback
            Err(BioDSLError::ParseError(format!(
                "Unknown robot_swarm method: {}",
                method
            )))
        } else {
            Err(BioDSLError::ParseError("Expected command".to_string()))
        }
    }

    /// Parse quantity unit
    fn parse_quantity_unit(&mut self) -> Result<QuantityUnit, BioDSLError> {
        if self.check(&Token::Milligrams) {
            self.advance();
            Ok(QuantityUnit::Milligrams)
        } else if self.check(&Token::Micrograms) {
            self.advance();
            Ok(QuantityUnit::Micrograms)
        } else if self.check(&Token::Nanograms) {
            self.advance();
            Ok(QuantityUnit::Nanograms)
        } else if self.check(&Token::Grams) {
            self.advance();
            Ok(QuantityUnit::Grams)
        } else if self.check(&Token::Moles) {
            self.advance();
            Ok(QuantityUnit::Moles)
        } else if self.check(&Token::Millimoles) {
            self.advance();
            Ok(QuantityUnit::Millimoles)
        } else if self.check(&Token::Micromoles) {
            self.advance();
            Ok(QuantityUnit::Micromoles)
        } else if let Some(Token::Identifier(s)) = self.peek_token() {
            let unit = match s.to_lowercase().as_str() {
                "mg" => QuantityUnit::Milligrams,
                "ug" => QuantityUnit::Micrograms,
                "ng" => QuantityUnit::Nanograms,
                "g" => QuantityUnit::Grams,
                "mol" => QuantityUnit::Moles,
                "mmol" => QuantityUnit::Millimoles,
                "umol" => QuantityUnit::Micromoles,
                _ => QuantityUnit::Milligrams,
            };
            self.advance();
            Ok(unit)
        } else {
            Ok(QuantityUnit::Milligrams)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // HELPER METHODS
    // ═══════════════════════════════════════════════════════════════════════════════

    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.position).map(|t| &t.token)
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.position += 1;
        }
        self.tokens.get(self.position - 1).map(|t| &t.token)
    }

    fn check(&self, token: &Token) -> bool {
        if let Some(current) = self.peek_token() {
            std::mem::discriminant(current) == std::mem::discriminant(token)
        } else {
            false
        }
    }

    fn check_identifier(&self) -> bool {
        matches!(self.peek_token(), Some(Token::Identifier(_)))
    }

    fn expect(&mut self, expected: &Token) -> Result<(), BioDSLError> {
        if self.check(expected) {
            self.advance();
            Ok(())
        } else {
            Err(BioDSLError::ParseError(format!(
                "Expected {:?}, got {:?}",
                expected,
                self.peek_token()
            )))
        }
    }

    fn expect_identifier(&mut self) -> Result<String, BioDSLError> {
        match self.peek_token() {
            Some(Token::Identifier(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            other => Err(BioDSLError::ParseError(format!(
                "Expected identifier, got {:?}",
                other
            ))),
        }
    }

    fn expect_string(&mut self) -> Result<String, BioDSLError> {
        match self.peek_token() {
            Some(Token::StringLiteral(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            other => Err(BioDSLError::ParseError(format!(
                "Expected string, got {:?}",
                other
            ))),
        }
    }

    fn expect_int(&mut self) -> Result<i64, BioDSLError> {
        match self.peek_token() {
            Some(Token::IntLiteral(n)) => {
                let n = *n;
                self.advance();
                Ok(n)
            }
            Some(Token::FloatLiteral(f)) => {
                let n = *f as i64;
                self.advance();
                Ok(n)
            }
            other => Err(BioDSLError::ParseError(format!(
                "Expected integer, got {:?}",
                other
            ))),
        }
    }

    fn expect_float(&mut self) -> Result<f64, BioDSLError> {
        match self.peek_token() {
            Some(Token::FloatLiteral(f)) => {
                let f = *f;
                self.advance();
                Ok(f)
            }
            Some(Token::IntLiteral(n)) => {
                let f = *n as f64;
                self.advance();
                Ok(f)
            }
            other => Err(BioDSLError::ParseError(format!(
                "Expected number, got {:?}",
                other
            ))),
        }
    }
}

impl Default for BioDSLParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_molecule() {
        let source = r#"
            molecule Water {
                smiles: "O"
            }
        "#;

        let parser = BioDSLParser::new();
        let program = parser.parse(source).unwrap();

        assert_eq!(program.definitions.len(), 1);
        if let Definition::Molecule(mol) = &program.definitions[0] {
            assert_eq!(mol.name, "Water");
            assert_eq!(mol.smiles, Some("O".to_string()));
        } else {
            panic!("Expected molecule definition");
        }
    }

    #[test]
    fn test_parse_genetic_circuit() {
        let source = r#"
            genetic_circuit ToggleSwitch {
                promoter pTet {
                    binding_site: tet_operator
                    strength: 0.8
                }
                gene lacI {
                    promoter: pTet
                    product: LacI_repressor
                    degrades_in: 30 minutes
                }
            }
        "#;

        let parser = BioDSLParser::new();
        let program = parser.parse(source).unwrap();

        assert_eq!(program.definitions.len(), 1);
        if let Definition::GeneticCircuit(circuit) = &program.definitions[0] {
            assert_eq!(circuit.name, "ToggleSwitch");
            assert_eq!(circuit.promoters.len(), 1);
            assert_eq!(circuit.genes.len(), 1);
        } else {
            panic!("Expected genetic circuit definition");
        }
    }

    #[test]
    fn test_parse_synthesize_command() {
        let source = r#"
            molecule THC {
                smiles: "CCCCCC"
            }
            synthesize(THC, amount: 1.0 mg, purity: 0.999)
        "#;

        let parser = BioDSLParser::new();
        let program = parser.parse(source).unwrap();

        assert_eq!(program.definitions.len(), 1);
        assert_eq!(program.commands.len(), 1);

        if let Command::Synthesize(cmd) = &program.commands[0] {
            assert_eq!(cmd.target, "THC");
            assert_eq!(cmd.purity, Some(0.999));
        } else {
            panic!("Expected synthesize command");
        }
    }
}
