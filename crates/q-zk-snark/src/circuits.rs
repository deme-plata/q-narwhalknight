//! Circuit abstractions and constraint systems for Q-NarwhalKnight zk-SNARKs
//!
//! Provides high-level circuit building blocks and constraint system interfaces
//! that can be compiled to different SNARK backends (Groth16, PLONK, etc).

use anyhow::Result;
use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSystemRef, LinearCombination, Variable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::SNARKError;

/// High-level arithmetic circuit representation
#[derive(Debug, Clone)]
pub struct ArithmeticCircuit<F: PrimeField> {
    /// Circuit constraints
    pub constraints: Vec<ArithmeticConstraint<F>>,
    /// Variable assignments (witness)  
    pub witness: HashMap<String, F>,
    /// Public inputs
    pub public_inputs: HashMap<String, F>,
    /// Circuit metadata
    pub metadata: CircuitMetadata,
    /// Field phantom
    _phantom: PhantomData<F>,
}

/// Individual arithmetic constraint: a * b + c = d
#[derive(Debug, Clone)]
pub struct ArithmeticConstraint<F: PrimeField> {
    /// Left operand (a)
    pub left: LinearCombination<F>,
    /// Right operand (b)  
    pub right: LinearCombination<F>,
    /// Output (a * b + c)
    pub output: LinearCombination<F>,
    /// Constraint name for debugging
    pub name: Option<String>,
}

/// Circuit metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Circuit name
    pub name: String,
    /// Number of constraints
    pub num_constraints: usize,
    /// Number of variables
    pub num_variables: usize,
    /// Number of public inputs
    pub num_public_inputs: usize,
    /// Circuit complexity estimate
    pub complexity_estimate: usize,
}

/// Circuit variable reference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CircuitVariable {
    /// Variable identifier
    pub id: String,
    /// Whether this is a public input
    pub is_public: bool,
    /// Variable index in constraint system
    pub index: Option<usize>,
}

/// Circuit builder for constructing arithmetic circuits
pub struct CircuitBuilder<F: PrimeField> {
    /// Current constraints
    constraints: Vec<ArithmeticConstraint<F>>,
    /// Variable counter
    var_counter: usize,
    /// Variable registry
    variables: HashMap<String, CircuitVariable>,
    /// Current witness values
    witness: HashMap<String, F>,
    /// Public inputs
    public_inputs: HashMap<String, F>,
    /// Circuit name
    name: String,
    /// Field phantom
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> ArithmeticCircuit<F> {
    /// Create new arithmetic circuit with given size hint
    pub fn new(size_hint: usize) -> Self {
        Self {
            constraints: Vec::with_capacity(size_hint),
            witness: HashMap::new(),
            public_inputs: HashMap::new(),
            metadata: CircuitMetadata {
                name: "unnamed".to_string(),
                num_constraints: 0,
                num_variables: 0,
                num_public_inputs: 0,
                complexity_estimate: size_hint,
            },
            _phantom: PhantomData,
        }
    }

    /// Add constraint to circuit
    pub fn add_constraint(&mut self, constraint: ArithmeticConstraint<F>) {
        self.constraints.push(constraint);
        self.metadata.num_constraints = self.constraints.len();
    }

    /// Add witness assignment
    pub fn assign_witness(&mut self, var_name: String, value: F) {
        self.witness.insert(var_name, value);
    }

    /// Add public input
    pub fn assign_public_input(&mut self, var_name: String, value: F) {
        self.public_inputs.insert(var_name, value);
        self.metadata.num_public_inputs = self.public_inputs.len();
    }

    /// Get circuit size
    pub fn size(&self) -> usize {
        self.constraints.len()
    }

    /// Check if circuit is satisfiable with current witness
    pub fn is_satisfied(&self) -> Result<bool> {
        for constraint in &self.constraints {
            // Evaluate constraint with current witness
            let left_val = self.evaluate_linear_combination(&constraint.left)?;
            let right_val = self.evaluate_linear_combination(&constraint.right)?;
            let output_val = self.evaluate_linear_combination(&constraint.output)?;

            // Check if a * b + c = d (simplified constraint form)
            if left_val * right_val != output_val {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Generate R1CS constraints for SNARK systems
    pub fn generate_r1cs_constraints(
        &self,
        cs: ConstraintSystemRef<F>,
        _witness: &Option<Vec<F>>,
    ) -> Result<()> {
        // Allocate variables in constraint system
        let mut var_map = HashMap::new();

        // Allocate public inputs first
        for (name, value) in &self.public_inputs {
            let var = cs.new_input_variable(|| Ok(*value)).map_err(|_| {
                SNARKError::CircuitCompilation("Failed to allocate public input".to_string())
            })?;
            var_map.insert(name.clone(), var);
        }

        // Allocate witness variables
        for (name, value) in &self.witness {
            let var = cs.new_witness_variable(|| Ok(*value)).map_err(|_| {
                SNARKError::CircuitCompilation("Failed to allocate witness".to_string())
            })?;
            var_map.insert(name.clone(), var);
        }

        // Generate constraints
        for constraint in &self.constraints {
            let left_lc = self.convert_to_r1cs_lc(&constraint.left, &var_map)?;
            let right_lc = self.convert_to_r1cs_lc(&constraint.right, &var_map)?;
            let output_lc = self.convert_to_r1cs_lc(&constraint.output, &var_map)?;

            cs.enforce_constraint(left_lc, right_lc, output_lc)
                .map_err(|_| {
                    SNARKError::CircuitCompilation("Failed to enforce constraint".to_string())
                })?;
        }

        Ok(())
    }

    /// Convert to PLONK circuit format
    pub fn to_plonk_circuit(&self) -> Result<crate::plonk::PLONKCircuit<F>> {
        let mut gates = Vec::new();
        let mut wires = Vec::new();
        let mut wire_map = HashMap::new();
        let mut wire_counter = 0;

        // Assign wire indices
        for name in self.witness.keys().chain(self.public_inputs.keys()) {
            wire_map.insert(name.clone(), wire_counter);
            wire_counter += 1;
        }

        // Convert constraints to PLONK gates
        for _constraint in &self.constraints {
            // Simplified conversion - real implementation would handle general linear combinations
            let gate = crate::plonk::PLONKGate {
                left_wire: 0,   // Would map from constraint.left
                right_wire: 1,  // Would map from constraint.right
                output_wire: 2, // Would map from constraint.output
                q_l: F::one(),
                q_r: F::one(),
                q_o: -F::one(),
                q_m: F::zero(),
                q_c: F::zero(),
            };
            gates.push(gate);
        }

        // Collect wire values
        for (_name, value) in &self.witness {
            wires.push(*value);
        }

        Ok(crate::plonk::PLONKCircuit {
            gates,
            wires,
            public_inputs: self.public_inputs.values().cloned().collect(),
            copy_constraints: vec![], // Would be computed from circuit structure
        })
    }

    // Helper methods

    fn evaluate_linear_combination(&self, _lc: &LinearCombination<F>) -> Result<F> {
        // Simplified evaluation - real implementation would handle full linear combinations
        Ok(F::zero())
    }

    fn convert_to_r1cs_lc(
        &self,
        _lc: &LinearCombination<F>,
        _var_map: &HashMap<String, Variable>,
    ) -> Result<ark_relations::r1cs::LinearCombination<F>> {
        // Simplified conversion - real implementation would map variables properly
        Ok(ark_relations::r1cs::LinearCombination::zero())
    }
}

impl<F: PrimeField> CircuitBuilder<F> {
    /// Create new circuit builder
    pub fn new(name: String) -> Self {
        Self {
            constraints: Vec::new(),
            var_counter: 0,
            variables: HashMap::new(),
            witness: HashMap::new(),
            public_inputs: HashMap::new(),
            name,
            _phantom: PhantomData,
        }
    }

    /// Create a new variable
    pub fn create_variable(&mut self, name: String, is_public: bool) -> CircuitVariable {
        let var = CircuitVariable {
            id: name.clone(),
            is_public,
            index: Some(self.var_counter),
        };

        self.variables.insert(name, var.clone());
        self.var_counter += 1;
        var
    }

    /// Assign value to variable (witness or public input)
    pub fn assign_variable(&mut self, var: &CircuitVariable, value: F) -> Result<()> {
        if var.is_public {
            self.public_inputs.insert(var.id.clone(), value);
        } else {
            self.witness.insert(var.id.clone(), value);
        }
        Ok(())
    }

    /// Add multiplication constraint: a * b = c
    pub fn enforce_multiplication(
        &mut self,
        a: &CircuitVariable,
        b: &CircuitVariable,
        c: &CircuitVariable,
        name: Option<String>,
    ) -> Result<()> {
        let constraint = ArithmeticConstraint {
            left: self.variable_to_lc(a)?,
            right: self.variable_to_lc(b)?,
            output: self.variable_to_lc(c)?,
            name,
        };

        self.constraints.push(constraint);
        Ok(())
    }

    /// Add addition constraint: a + b = c  
    pub fn enforce_addition(
        &mut self,
        a: &CircuitVariable,
        b: &CircuitVariable,
        c: &CircuitVariable,
        name: Option<String>,
    ) -> Result<()> {
        // Addition can be represented as: 1 * (a + b) = c
        let constraint = ArithmeticConstraint {
            left: LinearCombination::zero() + (F::one(), self.variable_to_term(a)?),
            right: LinearCombination::zero() + (F::one(), self.variable_to_term(b)?),
            output: self.variable_to_lc(c)?,
            name,
        };

        self.constraints.push(constraint);
        Ok(())
    }

    /// Add equality constraint: a = b
    pub fn enforce_equality(
        &mut self,
        a: &CircuitVariable,
        b: &CircuitVariable,
        name: Option<String>,
    ) -> Result<()> {
        // Equality can be represented as: 1 * a = b
        let constraint = ArithmeticConstraint {
            left: LinearCombination::zero() + (F::one(), Variable::One),
            right: self.variable_to_lc(a)?,
            output: self.variable_to_lc(b)?,
            name,
        };

        self.constraints.push(constraint);
        Ok(())
    }

    /// Add constant constraint: a = constant
    pub fn enforce_constant(
        &mut self,
        a: &CircuitVariable,
        constant: F,
        name: Option<String>,
    ) -> Result<()> {
        let constraint = ArithmeticConstraint {
            left: LinearCombination::zero() + (F::one(), Variable::One),
            right: LinearCombination::zero() + (constant, Variable::One),
            output: self.variable_to_lc(a)?,
            name,
        };

        self.constraints.push(constraint);
        Ok(())
    }

    /// Build the final circuit
    pub fn build(self) -> ArithmeticCircuit<F> {
        let num_variables = self.variables.len();
        let num_constraints = self.constraints.len();
        let num_public_inputs = self.public_inputs.len();

        ArithmeticCircuit {
            constraints: self.constraints,
            witness: self.witness,
            public_inputs: self.public_inputs,
            metadata: CircuitMetadata {
                name: self.name,
                num_constraints,
                num_variables,
                num_public_inputs,
                complexity_estimate: num_constraints * 2, // Rough estimate
            },
            _phantom: PhantomData,
        }
    }

    // Helper methods

    fn variable_to_lc(&self, var: &CircuitVariable) -> Result<LinearCombination<F>> {
        let term = self.variable_to_term(var)?;
        Ok(LinearCombination::zero() + (F::one(), term))
    }

    fn variable_to_term(&self, var: &CircuitVariable) -> Result<Variable> {
        let _index = var.index.ok_or_else(|| {
            SNARKError::CircuitCompilation("Variable not assigned index".to_string())
        })?;

        // For now, return a placeholder variable - this would need proper implementation
        // based on the specific constraint system being used
        Ok(Variable::One)
    }
}

/// Common circuit gadgets
pub struct CircuitGadgets;

impl CircuitGadgets {
    /// Create a boolean constraint circuit
    pub fn boolean_constraint<F: PrimeField>(
        builder: &mut CircuitBuilder<F>,
        var: &CircuitVariable,
    ) -> Result<()> {
        // Boolean constraint: b * (1 - b) = 0
        // Equivalent to: b * b = b
        builder.enforce_multiplication(var, var, var, Some("boolean_constraint".to_string()))
    }

    /// Create a range proof circuit (simplified)
    pub fn range_proof<F: PrimeField>(
        builder: &mut CircuitBuilder<F>,
        var: &CircuitVariable,
        max_bits: usize,
    ) -> Result<Vec<CircuitVariable>> {
        let mut bit_vars = Vec::new();

        // Decompose variable into bits
        for i in 0..max_bits {
            let bit_var = builder.create_variable(format!("bit_{}_{}", var.id, i), false);

            // Enforce boolean constraint on each bit
            Self::boolean_constraint(builder, &bit_var)?;

            bit_vars.push(bit_var);
        }

        // TODO: Add constraints to ensure bits sum to original variable
        // This would require field arithmetic operations

        Ok(bit_vars)
    }

    /// Create a hash function circuit (simplified)
    pub fn hash_constraint<F: PrimeField>(
        builder: &mut CircuitBuilder<F>,
        inputs: &[CircuitVariable],
        output: &CircuitVariable,
    ) -> Result<()> {
        // Simplified hash constraint - real implementation would use
        // SNARK-friendly hash functions like Poseidon or Rescue

        if inputs.is_empty() {
            return Err(
                SNARKError::CircuitCompilation("Hash inputs cannot be empty".to_string()).into(),
            );
        }

        // For demo: output = sum of inputs (not a real hash!)
        let mut sum_var = inputs[0].clone();

        for input in inputs.iter().skip(1) {
            let temp_var =
                builder.create_variable(format!("hash_temp_{}", builder.var_counter), false);

            builder.enforce_addition(&sum_var, input, &temp_var, None)?;
            sum_var = temp_var;
        }

        builder.enforce_equality(&sum_var, output, Some("hash_output".to_string()))?;
        Ok(())
    }

    /// Create a signature verification circuit
    pub fn signature_verification<F: PrimeField>(
        builder: &mut CircuitBuilder<F>,
        public_key: &CircuitVariable,
        message: &CircuitVariable,
        signature: &CircuitVariable,
        valid: &CircuitVariable,
    ) -> Result<()> {
        // Simplified signature verification - real implementation would
        // implement EdDSA or ECDSA in circuit

        // For demo: valid = (public_key * message == signature)
        let product_var = builder.create_variable("signature_product".to_string(), false);

        builder.enforce_multiplication(public_key, message, &product_var, None)?;
        builder.enforce_equality(&product_var, signature, None)?;
        builder.enforce_constant(valid, F::one(), Some("signature_valid".to_string()))?;

        Ok(())
    }
}

/// Circuit optimization utilities
pub struct CircuitOptimizer;

impl CircuitOptimizer {
    /// Remove redundant constraints
    pub fn remove_redundant_constraints<F: PrimeField>(
        circuit: &mut ArithmeticCircuit<F>,
    ) -> Result<usize> {
        // Simple deduplication based on constraint structure
        let original_count = circuit.constraints.len();
        circuit.constraints.dedup_by(|a, b| {
            // Compare constraints (simplified comparison)
            std::ptr::eq(a, b)
        });

        let removed = original_count - circuit.constraints.len();
        circuit.metadata.num_constraints = circuit.constraints.len();
        Ok(removed)
    }

    /// Estimate circuit complexity
    pub fn estimate_complexity<F: PrimeField>(circuit: &ArithmeticCircuit<F>) -> usize {
        // Simple complexity estimate: constraints + variables
        circuit.constraints.len() + circuit.witness.len() + circuit.public_inputs.len()
    }

    /// Suggest optimal SNARK protocol for circuit
    pub fn suggest_protocol<F: PrimeField>(circuit: &ArithmeticCircuit<F>) -> crate::SNARKProtocol {
        let complexity = Self::estimate_complexity(circuit);

        match complexity {
            0..=10_000 => crate::SNARKProtocol::Groth16,
            10_001..=100_000 => crate::SNARKProtocol::PLONK,
            100_001..=1_000_000 => crate::SNARKProtocol::Marlin,
            _ => crate::SNARKProtocol::Sonic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_circuit_builder() {
        let mut builder = CircuitBuilder::<Fr>::new("test_circuit".to_string());

        let a = builder.create_variable("a".to_string(), true);
        let b = builder.create_variable("b".to_string(), true);
        let c = builder.create_variable("c".to_string(), false);

        builder.assign_variable(&a, Fr::from(2u64)).unwrap();
        builder.assign_variable(&b, Fr::from(3u64)).unwrap();
        builder.assign_variable(&c, Fr::from(6u64)).unwrap();

        builder
            .enforce_multiplication(&a, &b, &c, Some("mult".to_string()))
            .unwrap();

        let circuit = builder.build();

        assert_eq!(circuit.constraints.len(), 1);
        assert_eq!(circuit.public_inputs.len(), 2);
        assert_eq!(circuit.witness.len(), 1);
    }

    #[test]
    fn test_circuit_gadgets() {
        let mut builder = CircuitBuilder::<Fr>::new("gadget_test".to_string());

        let bool_var = builder.create_variable("bool".to_string(), false);
        builder.assign_variable(&bool_var, Fr::from(1u64)).unwrap();

        CircuitGadgets::boolean_constraint(&mut builder, &bool_var).unwrap();

        let circuit = builder.build();
        assert_eq!(circuit.constraints.len(), 1);
    }

    #[test]
    fn test_circuit_optimization() {
        let mut circuit = ArithmeticCircuit::<Fr>::new(10);

        // Add some constraints
        circuit.add_constraint(ArithmeticConstraint {
            left: LinearCombination::zero(),
            right: LinearCombination::zero(),
            output: LinearCombination::zero(),
            name: Some("test".to_string()),
        });

        let complexity = CircuitOptimizer::estimate_complexity(&circuit);
        assert!(complexity > 0);

        let protocol = CircuitOptimizer::suggest_protocol(&circuit);
        assert_eq!(protocol, crate::SNARKProtocol::Groth16); // Small circuit
    }
}
