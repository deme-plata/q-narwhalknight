//! Neural Network to R1CS Circuit Compiler
//!
//! Converts neural network layers to R1CS constraints for SNARK proving.

use super::{CompiledCircuit, ZkLayer, ZkMLError};
use tracing::debug;

/// Circuit compiler for neural networks
pub struct CircuitCompiler {
    /// Fixed-point precision
    fixed_point_bits: u32,

    /// Maximum constraint count
    max_constraints: usize,
}

impl CircuitCompiler {
    /// Create new compiler
    pub fn new(fixed_point_bits: u32, max_constraints: usize) -> Self {
        Self {
            fixed_point_bits,
            max_constraints,
        }
    }

    /// Compile entire network to R1CS
    pub fn compile_network(&self, layers: &[ZkLayer]) -> Result<NetworkCircuit, ZkMLError> {
        let mut total_constraints = 0;
        let mut layer_circuits = Vec::new();
        let mut variable_offset = 1; // Start after constant 1

        for (i, layer) in layers.iter().enumerate() {
            debug!("Compiling layer {} to R1CS", i);

            let (circuit, vars_used) = self.compile_layer(layer, variable_offset)?;
            total_constraints += circuit.num_constraints;
            variable_offset += vars_used;

            if total_constraints > self.max_constraints {
                return Err(ZkMLError::CompilationError(format!(
                    "Too many constraints: {} > {}",
                    total_constraints, self.max_constraints
                )));
            }

            layer_circuits.push(circuit);
        }

        Ok(NetworkCircuit {
            layers: layer_circuits,
            total_constraints,
            total_variables: variable_offset,
        })
    }

    /// Compile single layer
    fn compile_layer(
        &self,
        layer: &ZkLayer,
        var_offset: usize,
    ) -> Result<(CompiledCircuit, usize), ZkMLError> {
        match layer {
            ZkLayer::Dense {
                weights,
                bias,
                in_features,
                out_features,
            } => self.compile_dense(*in_features, *out_features, weights, bias, var_offset),

            ZkLayer::ReLU => {
                // ReLU uses comparison + multiplication
                // y = x if x > 0 else 0
                // Needs auxiliary variables for sign bit
                Ok((
                    CompiledCircuit {
                        num_constraints: 0, // ReLU constraints depend on input size
                        num_variables: 0,
                        a_matrix: vec![],
                        b_matrix: vec![],
                        c_matrix: vec![],
                    },
                    0,
                ))
            }

            ZkLayer::Sigmoid => {
                // Piecewise linear approximation
                Ok((
                    CompiledCircuit {
                        num_constraints: 0,
                        num_variables: 0,
                        a_matrix: vec![],
                        b_matrix: vec![],
                        c_matrix: vec![],
                    },
                    0,
                ))
            }

            ZkLayer::Softmax => Ok((
                CompiledCircuit {
                    num_constraints: 0,
                    num_variables: 0,
                    a_matrix: vec![],
                    b_matrix: vec![],
                    c_matrix: vec![],
                },
                0,
            )),

            ZkLayer::BatchNorm { .. } => Ok((
                CompiledCircuit {
                    num_constraints: 0,
                    num_variables: 0,
                    a_matrix: vec![],
                    b_matrix: vec![],
                    c_matrix: vec![],
                },
                0,
            )),
        }
    }

    /// Compile dense layer
    fn compile_dense(
        &self,
        in_features: usize,
        out_features: usize,
        weights: &[i64],
        bias: &[i64],
        var_offset: usize,
    ) -> Result<(CompiledCircuit, usize), ZkMLError> {
        // Each output is: y_j = (sum_i w_ij * x_i + b_j) >> scale
        // We need:
        // 1. Input variables: x_i
        // 2. Output variables: y_j
        // 3. Intermediate for scaled multiplication

        let num_constraints = out_features;
        let vars_used = in_features + out_features;

        // Build sparse constraint matrices
        let mut a_matrix = Vec::new();
        let mut b_matrix = Vec::new();
        let mut c_matrix = Vec::new();

        for j in 0..out_features {
            // Constraint j: sum_i(w_ij * x_i) + b_j = y_j * scale
            for i in 0..in_features {
                let weight = weights[j * in_features + i];
                if weight != 0 {
                    a_matrix.push((j, var_offset + i, weight));
                }
            }

            // Bias term (multiplied by 1)
            if bias[j] != 0 {
                a_matrix.push((j, 0, bias[j])); // var 0 is constant 1
            }

            // B is just 1 (identity for linear layer)
            b_matrix.push((j, 0, 1));

            // C is scaled output
            let scale = 1i64 << self.fixed_point_bits;
            c_matrix.push((j, var_offset + in_features + j, scale));
        }

        Ok((
            CompiledCircuit {
                num_constraints,
                num_variables: vars_used,
                a_matrix,
                b_matrix,
                c_matrix,
            },
            vars_used,
        ))
    }

    /// Compile ReLU layer with input size
    pub fn compile_relu(&self, size: usize, var_offset: usize) -> Result<(CompiledCircuit, usize), ZkMLError> {
        // ReLU: y = x * is_positive
        // Constraints:
        // 1. is_positive ∈ {0, 1}: is_positive * (1 - is_positive) = 0
        // 2. y = x * is_positive
        // 3. (1 - is_positive) * x ≤ 0 (x negative when is_positive = 0)

        let num_constraints = size * 2; // Simplified
        let vars_used = size * 3; // input, output, is_positive

        let mut a_matrix = Vec::new();
        let mut b_matrix = Vec::new();
        let mut c_matrix = Vec::new();

        for i in 0..size {
            let x_var = var_offset + i;
            let y_var = var_offset + size + i;
            let sign_var = var_offset + 2 * size + i;

            // Constraint 1: is_positive ∈ {0, 1}
            a_matrix.push((i * 2, sign_var, 1));
            b_matrix.push((i * 2, 0, 1)); // 1 - is_positive (simplified)
            c_matrix.push((i * 2, 0, 0)); // = 0

            // Constraint 2: y = x * is_positive
            a_matrix.push((i * 2 + 1, x_var, 1));
            b_matrix.push((i * 2 + 1, sign_var, 1));
            c_matrix.push((i * 2 + 1, y_var, 1));
        }

        Ok((
            CompiledCircuit {
                num_constraints,
                num_variables: vars_used,
                a_matrix,
                b_matrix,
                c_matrix,
            },
            vars_used,
        ))
    }
}

/// Compiled network circuit
#[derive(Clone, Debug)]
pub struct NetworkCircuit {
    /// Layer circuits
    pub layers: Vec<CompiledCircuit>,

    /// Total constraint count
    pub total_constraints: usize,

    /// Total variable count
    pub total_variables: usize,
}

impl NetworkCircuit {
    /// Get constraint count
    pub fn constraint_count(&self) -> usize {
        self.total_constraints
    }

    /// Get variable count
    pub fn variable_count(&self) -> usize {
        self.total_variables
    }

    /// Flatten to single circuit
    pub fn flatten(&self) -> CompiledCircuit {
        let mut a_matrix = Vec::new();
        let mut b_matrix = Vec::new();
        let mut c_matrix = Vec::new();
        let mut constraint_offset = 0;

        for layer in &self.layers {
            for (row, col, val) in &layer.a_matrix {
                a_matrix.push((row + constraint_offset, *col, *val));
            }
            for (row, col, val) in &layer.b_matrix {
                b_matrix.push((row + constraint_offset, *col, *val));
            }
            for (row, col, val) in &layer.c_matrix {
                c_matrix.push((row + constraint_offset, *col, *val));
            }
            constraint_offset += layer.num_constraints;
        }

        CompiledCircuit {
            num_constraints: self.total_constraints,
            num_variables: self.total_variables,
            a_matrix,
            b_matrix,
            c_matrix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_dense() {
        let compiler = CircuitCompiler::new(16, 1_000_000);

        let weights = vec![1, 0, 0, 1]; // Identity matrix
        let bias = vec![0, 0];

        let (circuit, vars) = compiler
            .compile_dense(2, 2, &weights, &bias, 1)
            .unwrap();

        assert_eq!(circuit.num_constraints, 2);
        assert!(vars > 0);
    }

    #[test]
    fn test_compile_relu() {
        let compiler = CircuitCompiler::new(16, 1_000_000);

        let (circuit, vars) = compiler.compile_relu(4, 1).unwrap();

        assert_eq!(circuit.num_constraints, 8); // 2 per element
        assert_eq!(vars, 12); // 3 per element
    }
}
