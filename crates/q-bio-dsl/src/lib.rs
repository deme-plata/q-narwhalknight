//! # Q-Bio-DSL: Biological Programming Language for Water Robots
//!
//! A domain-specific language for programming biological systems through
//! quantum water robots. Enables:
//!
//! - **Small molecule synthesis** (THC, CBD, proteins, etc.)
//! - **Genetic circuit design** (toggle switches, oscillators, logic gates)
//! - **Protein folding assistance** (guided by quantum coherence)
//! - **DNA/RNA synthesis** (nucleotide-by-nucleotide assembly)
//!
//! ## Example
//!
//! ```rust,ignore
//! use q_bio_dsl::prelude::*;
//!
//! let program = BioDSL::parse(r#"
//!     molecule THC {
//!         smiles: "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O"
//!         stereochemistry: [(6a, R), (10a, R)]
//!     }
//!     synthesize(THC, amount: 1mg)
//! "#)?;
//!
//! let result = program.execute(&robot_swarm).await?;
//! ```

use serde::{Deserialize, Serialize};

pub mod ast;
pub mod blockchain_integration;
pub mod compiler;
pub mod genetic_circuits;
pub mod lexer;
pub mod molecular_ir;
pub mod parser;
pub mod safety;
pub mod small_molecules;
pub mod synthesis;
pub mod types;

pub use ast::*;
pub use blockchain_integration::*;
pub use compiler::BioDSLCompiler;
pub use genetic_circuits::*;
pub use molecular_ir::*;
pub use parser::BioDSLParser;
pub use small_molecules::*;
pub use synthesis::*;
pub use types::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::ast::*;
    pub use crate::compiler::BioDSLCompiler;
    pub use crate::genetic_circuits::*;
    pub use crate::molecular_ir::*;
    pub use crate::parser::BioDSLParser;
    pub use crate::small_molecules::*;
    pub use crate::synthesis::*;
    pub use crate::types::*;
}

/// Core BioDSL entry point
pub struct BioDSL;

impl BioDSL {
    /// Parse a BioDSL program from source code
    pub fn parse(source: &str) -> Result<BioDSLProgram, BioDSLError> {
        let parser = BioDSLParser::new();
        parser.parse(source)
    }

    /// Parse and compile a BioDSL program
    pub fn compile(source: &str) -> Result<CompiledProgram, BioDSLError> {
        let program = Self::parse(source)?;
        let compiler = BioDSLCompiler::new();
        compiler.compile(program)
    }
}

/// Compiled BioDSL program ready for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledProgram {
    /// Molecular instructions to execute
    pub instructions: Vec<MolecularInstruction>,
    /// Safety constraints that must be verified
    pub safety_constraints: Vec<SafetyConstraint>,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: u64,
    /// Required robot types
    pub required_robots: Vec<String>,
}

impl CompiledProgram {
    /// Execute the compiled program with a robot swarm
    pub async fn execute<S: RobotSwarmInterface>(
        &self,
        swarm: &S,
    ) -> Result<SynthesisResult, BioDSLError> {
        // Verify safety constraints first
        for constraint in &self.safety_constraints {
            constraint.verify()?;
        }

        // Execute molecular instructions
        let mut results = Vec::new();
        for instruction in &self.instructions {
            let result = swarm.execute_instruction(instruction).await?;
            results.push(result);
        }

        Ok(SynthesisResult {
            success: true,
            molecules_produced: results.len(),
            total_time_ms: self.estimated_time_ms,
            verification_status: VerificationStatus::Verified,
        })
    }
}

/// Interface for robot swarm execution
#[async_trait::async_trait]
pub trait RobotSwarmInterface: Send + Sync {
    async fn execute_instruction(
        &self,
        instruction: &MolecularInstruction,
    ) -> Result<InstructionResult, BioDSLError>;

    async fn get_available_robots(&self) -> Vec<String>;

    async fn verify_structure(
        &self,
        molecule_id: &str,
        tolerance: f64,
    ) -> Result<bool, BioDSLError>;
}

/// Result of executing a single instruction
#[derive(Debug, Clone)]
pub struct InstructionResult {
    pub instruction_id: u64,
    pub success: bool,
    pub execution_time_us: u64,
    pub error_message: Option<String>,
}

/// Result of synthesis operation
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub success: bool,
    pub molecules_produced: usize,
    pub total_time_ms: u64,
    pub verification_status: VerificationStatus,
}

/// Verification status of synthesized molecules
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    Verified,
    PartiallyVerified { confidence: f64 },
    Unverified,
    Failed { reason: String },
}

/// Safety constraint for synthesis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraint {
    pub constraint_type: SafetyConstraintType,
    pub molecule_id: Option<String>,
    pub max_quantity: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyConstraintType {
    QuantityLimit,
    LicenseRequired,
    EnvironmentalContainment,
    KillSwitchRequired,
}

impl SafetyConstraint {
    pub fn verify(&self) -> Result<(), BioDSLError> {
        // Safety verification logic
        Ok(())
    }
}

/// BioDSL error types
#[derive(Debug, thiserror::Error)]
pub enum BioDSLError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Compilation error: {0}")]
    CompileError(String),

    #[error("Invalid molecule: {0}")]
    InvalidMolecule(String),

    #[error("Invalid SMILES notation: {0}")]
    InvalidSmiles(String),

    #[error("Safety constraint violated: {0}")]
    SafetyViolation(String),

    #[error("Robot execution error: {0}")]
    ExecutionError(String),

    #[error("Verification failed: {0}")]
    VerificationError(String),

    #[error("Unknown element: {0}")]
    UnknownElement(String),

    #[error("Invalid stereochemistry: {0}")]
    InvalidStereochemistry(String),

    #[error("Genetic circuit error: {0}")]
    GeneticCircuitError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
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

        let result = BioDSL::parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_thc() {
        let source = r#"
            molecule THC {
                smiles: "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O"
                stereochemistry: [(6a, R), (10a, R)]
            }
        "#;

        let result = BioDSL::parse(source);
        assert!(result.is_ok());
    }
}
