//! AIR (Algebraic Intermediate Representation) for STARK Constraints
//!
//! This module defines the constraint system for STARK proofs, including
//! execution trace format and arithmetic constraint definitions.

use std::collections::HashMap;

/// Execution trace for STARK proving
#[derive(Clone, Debug)]
pub struct ExecutionTrace {
    /// Trace data organized as rows (time steps) and columns (registers)
    pub trace_matrix: Vec<Vec<u64>>,
    /// Number of time steps in the trace
    pub trace_length: usize,
    /// Number of registers/columns
    pub register_count: usize,
    /// Public inputs (subset of first trace row)
    pub public_inputs: Vec<u64>,
    /// Metadata for trace interpretation
    pub metadata: TraceMetadata,
}

impl ExecutionTrace {
    /// Create new execution trace
    pub fn new(trace_matrix: Vec<Vec<u64>>, public_inputs: Vec<u64>) -> Self {
        let trace_length = trace_matrix.len();
        let register_count = trace_matrix.first().map_or(0, |row| row.len());

        Self {
            trace_matrix,
            trace_length,
            register_count,
            public_inputs,
            metadata: TraceMetadata::default(),
        }
    }

    /// Validate trace structure
    pub fn validate(&self) -> Result<(), String> {
        if self.trace_matrix.is_empty() {
            return Err("Empty execution trace".to_string());
        }

        // Check all rows have same length
        for (i, row) in self.trace_matrix.iter().enumerate() {
            if row.len() != self.register_count {
                return Err(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    self.register_count
                ));
            }
        }

        // Check trace length is power of 2 (required for FFT)
        if !self.trace_length.is_power_of_two() {
            return Err(format!(
                "Trace length {} must be power of 2",
                self.trace_length
            ));
        }

        Ok(())
    }

    /// Get trace value at specific position
    pub fn get_value(&self, step: usize, register: usize) -> Option<u64> {
        self.trace_matrix.get(step)?.get(register).copied()
    }

    /// Extend trace with padding if needed
    pub fn pad_to_power_of_two(&mut self) {
        let next_power_of_two = self.trace_length.next_power_of_two();
        if next_power_of_two > self.trace_length {
            // Pad with last row repeated
            if let Some(last_row) = self.trace_matrix.last().cloned() {
                for _ in self.trace_length..next_power_of_two {
                    self.trace_matrix.push(last_row.clone());
                }
                self.trace_length = next_power_of_two;
            }
        }
    }
}

/// Metadata for execution trace
#[derive(Clone, Debug, Default)]
pub struct TraceMetadata {
    /// Description of what each register represents
    pub register_names: HashMap<usize, String>,
    /// Special register indices (PC, stack pointer, etc.)
    pub special_registers: HashMap<String, usize>,
    /// Memory layout information
    pub memory_layout: MemoryLayout,
}

/// Memory layout for trace execution
#[derive(Clone, Debug, Default)]
pub struct MemoryLayout {
    /// Program counter register index
    pub pc_register: Option<usize>,
    /// Stack pointer register index
    pub stack_pointer: Option<usize>,
    /// General purpose register range
    pub gp_register_range: Option<(usize, usize)>,
    /// Memory word size in bytes
    pub word_size: usize,
}

/// AIR constraint system for STARK proofs
#[derive(Clone, Debug)]
pub struct AirConstraints {
    /// Boundary constraints (initial and final states)
    pub boundary_constraints: Vec<BoundaryConstraint>,
    /// Transition constraints (between consecutive steps)
    pub transition_constraints: Vec<TransitionConstraint>,
    /// Global constraints (hold for entire execution)
    pub global_constraints: Vec<GlobalConstraint>,
    /// Constraint degree (maximum degree of polynomials)
    pub constraint_degree: usize,
    /// Blowup factor for low-degree testing
    pub blowup_factor: usize,
}

impl AirConstraints {
    /// Create new AIR constraint system
    pub fn new() -> Self {
        Self {
            boundary_constraints: Vec::new(),
            transition_constraints: Vec::new(),
            global_constraints: Vec::new(),
            constraint_degree: 2, // Default quadratic constraints
            blowup_factor: 8,     // Default 8x blowup
        }
    }

    /// Add boundary constraint
    pub fn add_boundary_constraint(&mut self, constraint: BoundaryConstraint) {
        self.boundary_constraints.push(constraint);
    }

    /// Add transition constraint
    pub fn add_transition_constraint(&mut self, constraint: TransitionConstraint) {
        // Update constraint degree if needed
        self.constraint_degree = self.constraint_degree.max(constraint.degree);
        self.transition_constraints.push(constraint);
    }

    /// Evaluate all constraints for a given trace
    pub fn evaluate_constraints(&self, trace: &ExecutionTrace) -> ConstraintEvaluations {
        let mut evaluations = ConstraintEvaluations::new();

        // Evaluate boundary constraints
        for constraint in &self.boundary_constraints {
            let result = constraint.evaluate(trace);
            evaluations.boundary_results.push(result);
        }

        // Evaluate transition constraints
        for constraint in &self.transition_constraints {
            for step in 0..(trace.trace_length - 1) {
                let result = constraint.evaluate_step(trace, step);
                evaluations.transition_results.push(result);
            }
        }

        // Evaluate global constraints
        for constraint in &self.global_constraints {
            let result = constraint.evaluate(trace);
            evaluations.global_results.push(result);
        }

        evaluations
    }

    /// Check if all constraints are satisfied
    pub fn verify_constraints(&self, trace: &ExecutionTrace) -> bool {
        let evaluations = self.evaluate_constraints(trace);
        evaluations.all_satisfied()
    }
}

impl Default for AirConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Boundary constraint (initial/final state)
#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    /// Register index to constrain
    pub register: usize,
    /// Step to apply constraint (0 for initial, trace_length-1 for final)
    pub step: BoundaryStep,
    /// Expected value
    pub value: u64,
    /// Constraint description
    pub description: String,
}

impl BoundaryConstraint {
    /// Evaluate boundary constraint
    pub fn evaluate(&self, trace: &ExecutionTrace) -> u64 {
        let step_index = match self.step {
            BoundaryStep::Initial => 0,
            BoundaryStep::Final => trace.trace_length - 1,
            BoundaryStep::Step(s) => s,
        };

        if let Some(actual_value) = trace.get_value(step_index, self.register) {
            // Constraint satisfied if actual == expected (difference is 0)
            actual_value.wrapping_sub(self.value)
        } else {
            1 // Constraint violated (non-zero)
        }
    }
}

/// Boundary constraint step type
#[derive(Clone, Debug)]
pub enum BoundaryStep {
    Initial,     // First step (step 0)
    Final,       // Last step (step trace_length-1)
    Step(usize), // Specific step
}

/// Transition constraint (between consecutive steps)
#[derive(Clone, Debug)]
pub struct TransitionConstraint {
    /// Constraint expression
    pub expression: TransitionExpression,
    /// Constraint degree
    pub degree: usize,
    /// Constraint description
    pub description: String,
}

impl TransitionConstraint {
    /// Evaluate transition constraint at specific step
    pub fn evaluate_step(&self, trace: &ExecutionTrace, step: usize) -> u64 {
        self.expression.evaluate(trace, step)
    }
}

/// Expression for transition constraints
#[derive(Clone, Debug)]
pub enum TransitionExpression {
    /// Register value at current step
    Current(usize),
    /// Register value at next step
    Next(usize),
    /// Constant value
    Constant(u64),
    /// Addition of two expressions
    Add(Box<TransitionExpression>, Box<TransitionExpression>),
    /// Subtraction of two expressions
    Sub(Box<TransitionExpression>, Box<TransitionExpression>),
    /// Multiplication of two expressions
    Mul(Box<TransitionExpression>, Box<TransitionExpression>),
}

impl TransitionExpression {
    /// Evaluate expression at given step
    pub fn evaluate(&self, trace: &ExecutionTrace, step: usize) -> u64 {
        match self {
            TransitionExpression::Current(reg) => trace.get_value(step, *reg).unwrap_or(0),
            TransitionExpression::Next(reg) => trace.get_value(step + 1, *reg).unwrap_or(0),
            TransitionExpression::Constant(val) => *val,
            TransitionExpression::Add(left, right) => left
                .evaluate(trace, step)
                .wrapping_add(right.evaluate(trace, step)),
            TransitionExpression::Sub(left, right) => left
                .evaluate(trace, step)
                .wrapping_sub(right.evaluate(trace, step)),
            TransitionExpression::Mul(left, right) => left
                .evaluate(trace, step)
                .wrapping_mul(right.evaluate(trace, step)),
        }
    }
}

/// Global constraint (holds for entire execution)
#[derive(Clone, Debug)]
pub struct GlobalConstraint {
    /// Global property to check
    pub property: GlobalProperty,
    /// Constraint description
    pub description: String,
}

impl GlobalConstraint {
    /// Evaluate global constraint
    pub fn evaluate(&self, trace: &ExecutionTrace) -> u64 {
        match &self.property {
            GlobalProperty::MonotonicRegister(reg) => {
                // Check if register values are monotonically increasing
                for step in 0..(trace.trace_length - 1) {
                    let current = trace.get_value(step, *reg).unwrap_or(0);
                    let next = trace.get_value(step + 1, *reg).unwrap_or(0);
                    if next < current {
                        return 1; // Violation
                    }
                }
                0 // Satisfied
            }
            GlobalProperty::RegisterSum(reg, expected_sum) => {
                // Check if sum of all register values equals expected
                let actual_sum: u64 = (0..trace.trace_length)
                    .map(|step| trace.get_value(step, *reg).unwrap_or(0))
                    .sum();
                actual_sum.wrapping_sub(*expected_sum)
            }
        }
    }
}

/// Global property types
#[derive(Clone, Debug)]
pub enum GlobalProperty {
    /// Register must be monotonically increasing
    MonotonicRegister(usize),
    /// Sum of register values must equal expected value
    RegisterSum(usize, u64),
}

/// Results of constraint evaluation
#[derive(Debug)]
pub struct ConstraintEvaluations {
    pub boundary_results: Vec<u64>,
    pub transition_results: Vec<u64>,
    pub global_results: Vec<u64>,
}

impl ConstraintEvaluations {
    fn new() -> Self {
        Self {
            boundary_results: Vec::new(),
            transition_results: Vec::new(),
            global_results: Vec::new(),
        }
    }

    /// Check if all constraints are satisfied (all evaluations are 0)
    pub fn all_satisfied(&self) -> bool {
        self.boundary_results.iter().all(|&x| x == 0)
            && self.transition_results.iter().all(|&x| x == 0)
            && self.global_results.iter().all(|&x| x == 0)
    }

    /// Count number of violated constraints
    pub fn violation_count(&self) -> usize {
        self.boundary_results.iter().filter(|&&x| x != 0).count()
            + self.transition_results.iter().filter(|&&x| x != 0).count()
            + self.global_results.iter().filter(|&&x| x != 0).count()
    }

    /// Get satisfaction rate as percentage
    pub fn satisfaction_rate(&self) -> f64 {
        let total =
            self.boundary_results.len() + self.transition_results.len() + self.global_results.len();

        if total == 0 {
            return 100.0;
        }

        let violations = self.violation_count();
        ((total - violations) as f64 / total as f64) * 100.0
    }
}

/// Builder for common AIR constraint patterns
pub struct AirConstraintBuilder;

impl AirConstraintBuilder {
    /// Create simple arithmetic sequence constraint: next = current + 1
    pub fn arithmetic_sequence(register: usize) -> TransitionConstraint {
        TransitionConstraint {
            expression: TransitionExpression::Sub(
                Box::new(TransitionExpression::Next(register)),
                Box::new(TransitionExpression::Add(
                    Box::new(TransitionExpression::Current(register)),
                    Box::new(TransitionExpression::Constant(1)),
                )),
            ),
            degree: 1,
            description: format!("Register {} forms arithmetic sequence", register),
        }
    }

    /// Create multiplication constraint: reg_c = reg_a * reg_b
    pub fn multiplication(reg_a: usize, reg_b: usize, reg_c: usize) -> TransitionConstraint {
        TransitionConstraint {
            expression: TransitionExpression::Sub(
                Box::new(TransitionExpression::Current(reg_c)),
                Box::new(TransitionExpression::Mul(
                    Box::new(TransitionExpression::Current(reg_a)),
                    Box::new(TransitionExpression::Current(reg_b)),
                )),
            ),
            degree: 2,
            description: format!(
                "Register {} = Register {} * Register {}",
                reg_c, reg_a, reg_b
            ),
        }
    }

    /// Create initial value constraint
    pub fn initial_value(register: usize, value: u64) -> BoundaryConstraint {
        BoundaryConstraint {
            register,
            step: BoundaryStep::Initial,
            value,
            description: format!("Register {} starts with value {}", register, value),
        }
    }

    /// Create final value constraint
    pub fn final_value(register: usize, value: u64) -> BoundaryConstraint {
        BoundaryConstraint {
            register,
            step: BoundaryStep::Final,
            value,
            description: format!("Register {} ends with value {}", register, value),
        }
    }
}
