//! Neural Architecture Search (NAS)
//!
//! Phase 4: Evolutionary and quantum-inspired optimization of neural network
//! architectures for on-chain prediction models.
//!
//! ## Key Features
//!
//! - Genetic algorithm for architecture evolution
//! - Quantum-inspired mutation operators
//! - Multi-objective optimization (accuracy vs proof cost)
//! - On-chain governance for architecture selection

use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use tracing::{debug, info, warn};

/// Neural Architecture Search configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NasConfig {
    /// Population size
    pub population_size: usize,

    /// Number of generations
    pub max_generations: usize,

    /// Mutation rate (0.0 - 1.0)
    pub mutation_rate: f64,

    /// Crossover rate (0.0 - 1.0)
    pub crossover_rate: f64,

    /// Elite size (top performers kept unchanged)
    pub elite_size: usize,

    /// Maximum network depth
    pub max_depth: usize,

    /// Maximum width per layer
    pub max_width: usize,

    /// Minimum width per layer
    pub min_width: usize,

    /// Available layer types
    pub layer_types: Vec<LayerType>,

    /// Fitness weights (accuracy, efficiency, proof_cost)
    pub fitness_weights: (f64, f64, f64),

    /// Tournament selection size
    pub tournament_size: usize,
}

impl Default for NasConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            elite_size: 5,
            max_depth: 10,
            max_width: 512,
            min_width: 8,
            layer_types: vec![
                LayerType::Dense,
                LayerType::ReLU,
                LayerType::Sigmoid,
                LayerType::BatchNorm,
                LayerType::Dropout,
            ],
            fitness_weights: (0.7, 0.2, 0.1),
            tournament_size: 5,
        }
    }
}

/// Available layer types for architecture search
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    BatchNorm,
    Dropout,
    Softmax,
    ResidualConnection,
}

/// Chromosome representing a neural network architecture
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Architecture {
    /// Unique identifier
    pub id: u64,

    /// Layer genes
    pub layers: Vec<LayerGene>,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Fitness score (computed after evaluation)
    pub fitness: Option<FitnessScore>,

    /// Generation this architecture was created
    pub generation: usize,

    /// Parent IDs (for lineage tracking)
    pub parents: Vec<u64>,
}

/// Gene encoding a single layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerGene {
    /// Layer type
    pub layer_type: LayerType,

    /// Output dimension (for Dense layers)
    pub output_dim: usize,

    /// Activation strength (0.0 - 1.0, affects initialization)
    pub activation_strength: f64,

    /// Dropout rate (for Dropout layers)
    pub dropout_rate: f64,

    /// Skip connection target (for ResidualConnection)
    pub skip_target: Option<usize>,
}

/// Multi-objective fitness score
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessScore {
    /// Accuracy on validation set
    pub accuracy: f64,

    /// Inference efficiency (1/latency)
    pub efficiency: f64,

    /// ZK proof cost (1/constraints)
    pub proof_cost: f64,

    /// Combined weighted score
    pub combined: f64,

    /// Pareto rank (for NSGA-II style selection)
    pub pareto_rank: usize,

    /// Crowding distance (for diversity)
    pub crowding_distance: f64,
}

impl PartialEq for FitnessScore {
    fn eq(&self, other: &Self) -> bool {
        (self.combined - other.combined).abs() < 1e-10
    }
}

impl Eq for FitnessScore {}

impl PartialOrd for FitnessScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitnessScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.combined
            .partial_cmp(&other.combined)
            .unwrap_or(Ordering::Equal)
    }
}

/// Neural Architecture Search engine
pub struct NasEngine {
    config: NasConfig,

    /// Current population
    population: Vec<Architecture>,

    /// Best architecture found
    best: Option<Architecture>,

    /// Hall of fame (top architectures across all generations)
    hall_of_fame: Vec<Architecture>,

    /// Generation counter
    generation: usize,

    /// ID counter
    next_id: u64,

    /// Statistics
    stats: NasStats,
}

/// NAS statistics
#[derive(Clone, Debug, Default)]
pub struct NasStats {
    pub total_evaluations: u64,
    pub total_mutations: u64,
    pub total_crossovers: u64,
    pub generations_run: usize,
    pub best_fitness_history: Vec<f64>,
    pub avg_fitness_history: Vec<f64>,
}

impl NasEngine {
    /// Create new NAS engine
    pub fn new(config: NasConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            best: None,
            hall_of_fame: Vec::new(),
            generation: 0,
            next_id: 0,
            stats: NasStats::default(),
        }
    }

    /// Initialize population with random architectures
    pub fn initialize(&mut self, input_dim: usize, output_dim: usize) {
        info!(
            "🧬 Initializing NAS population: {} individuals, input_dim={}, output_dim={}",
            self.config.population_size, input_dim, output_dim
        );

        let mut rng = rand::thread_rng();
        self.population.clear();

        for _ in 0..self.config.population_size {
            let arch = self.generate_random_architecture(input_dim, output_dim, &mut rng);
            self.population.push(arch);
        }

        info!("✅ Population initialized");
    }

    /// Generate random architecture
    fn generate_random_architecture<R: Rng>(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Architecture {
        let num_layers = rng.gen_range(2..=self.config.max_depth);
        let mut layers = Vec::with_capacity(num_layers);

        let mut current_dim = input_dim;

        for i in 0..num_layers {
            let is_last = i == num_layers - 1;

            // Choose layer type
            let layer_type = if is_last {
                LayerType::Dense // Last layer must be Dense for output
            } else {
                // Random layer type with bias toward Dense + activation pattern
                let rand_val: f64 = rng.gen();
                if rand_val < 0.4 {
                    LayerType::Dense
                } else if rand_val < 0.7 {
                    LayerType::ReLU
                } else if rand_val < 0.85 {
                    LayerType::BatchNorm
                } else {
                    *self.config.layer_types.choose(rng).unwrap_or(&LayerType::Dense)
                }
            };

            // Determine output dimension
            let next_dim = if is_last {
                output_dim
            } else if layer_type == LayerType::Dense {
                rng.gen_range(self.config.min_width..=self.config.max_width)
            } else {
                current_dim
            };

            let gene = LayerGene {
                layer_type,
                output_dim: next_dim,
                activation_strength: rng.gen(),
                dropout_rate: if layer_type == LayerType::Dropout {
                    rng.gen_range(0.1..=0.5)
                } else {
                    0.0
                },
                skip_target: None,
            };

            if layer_type == LayerType::Dense {
                current_dim = next_dim;
            }

            layers.push(gene);
        }

        let id = self.next_id;
        self.next_id += 1;

        Architecture {
            id,
            layers,
            input_dim,
            output_dim,
            fitness: None,
            generation: self.generation,
            parents: vec![],
        }
    }

    /// Run evolution for one generation
    pub fn evolve<F>(&mut self, evaluator: F) -> Result<(), NasError>
    where
        F: Fn(&Architecture) -> FitnessScore,
    {
        info!("🧬 Generation {}: Evolving population", self.generation);

        // Evaluate fitness
        for arch in &mut self.population {
            if arch.fitness.is_none() {
                let score = evaluator(arch);
                arch.fitness = Some(score);
                self.stats.total_evaluations += 1;
            }
        }

        // Compute Pareto ranks and crowding distances (NSGA-II)
        self.compute_pareto_ranks();

        // Sort by Pareto rank first, then crowding distance (NSGA-II ordering)
        self.population.sort_by(|a, b| {
            let fa = a.fitness.as_ref().unwrap();
            let fb = b.fitness.as_ref().unwrap();

            // Lower Pareto rank is better
            match fa.pareto_rank.cmp(&fb.pareto_rank) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => {
                    // Same rank: higher crowding distance is better (more diverse)
                    fb.crowding_distance
                        .partial_cmp(&fa.crowding_distance)
                        .unwrap_or(Ordering::Equal)
                }
            }
        });

        // Update best
        if let Some(top) = self.population.first() {
            if self.best.is_none()
                || top.fitness.as_ref().unwrap().combined
                    > self.best.as_ref().unwrap().fitness.as_ref().unwrap().combined
            {
                self.best = Some(top.clone());
                info!(
                    "  🏆 New best: fitness={:.4}",
                    top.fitness.as_ref().unwrap().combined
                );
            }
        }

        // Update hall of fame
        self.update_hall_of_fame();

        // Record statistics
        let best_fitness = self.population.first()
            .and_then(|a| a.fitness.as_ref())
            .map(|f| f.combined)
            .unwrap_or(0.0);
        let avg_fitness = self.population.iter()
            .filter_map(|a| a.fitness.as_ref())
            .map(|f| f.combined)
            .sum::<f64>() / self.population.len() as f64;

        self.stats.best_fitness_history.push(best_fitness);
        self.stats.avg_fitness_history.push(avg_fitness);

        // Create next generation
        let mut next_population = Vec::with_capacity(self.config.population_size);
        let mut rng = rand::thread_rng();

        // Elitism: keep top performers
        for arch in self.population.iter().take(self.config.elite_size) {
            let mut elite = arch.clone();
            elite.generation = self.generation + 1;
            next_population.push(elite);
        }

        // Fill rest with offspring
        while next_population.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_select(&mut rng);
            let parent2 = self.tournament_select(&mut rng);

            // Crossover
            let mut child = if rng.gen::<f64>() < self.config.crossover_rate {
                self.crossover(&parent1, &parent2, &mut rng)
            } else {
                let mut c = parent1.clone();
                c.id = self.next_id;
                self.next_id += 1;
                c.parents = vec![parent1.id];
                c.fitness = None;
                c.generation = self.generation + 1;
                c
            };

            // Mutation
            if rng.gen::<f64>() < self.config.mutation_rate {
                self.mutate(&mut child, &mut rng);
                self.stats.total_mutations += 1;
            }

            next_population.push(child);
        }

        self.population = next_population;
        self.generation += 1;
        self.stats.generations_run += 1;

        Ok(())
    }

    /// Tournament selection with NSGA-II style comparison
    /// Uses Pareto rank first, then crowding distance for tie-breaking
    fn tournament_select<R: Rng>(&self, rng: &mut R) -> Architecture {
        let mut best: Option<&Architecture> = None;

        for _ in 0..self.config.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];

            let is_better = match (best, candidate.fitness.as_ref()) {
                (None, Some(_)) => true,
                (Some(current_best), Some(candidate_fitness)) => {
                    let best_fitness = current_best.fitness.as_ref().unwrap();

                    // NSGA-II comparison: lower Pareto rank is better
                    if candidate_fitness.pareto_rank < best_fitness.pareto_rank {
                        true
                    } else if candidate_fitness.pareto_rank == best_fitness.pareto_rank {
                        // Same rank: prefer higher crowding distance (more diverse)
                        candidate_fitness.crowding_distance > best_fitness.crowding_distance
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if is_better {
                best = Some(candidate);
            }
        }

        best.unwrap().clone()
    }

    /// Compute Pareto ranks for the entire population (NSGA-II fast non-dominated sort)
    fn compute_pareto_ranks(&mut self) {
        let n = self.population.len();
        if n == 0 {
            return;
        }

        // domination_count[i] = number of solutions that dominate solution i
        let mut domination_count = vec![0usize; n];
        // dominated_set[i] = set of solutions that solution i dominates
        let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Compare all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let fi = self.population[i].fitness.as_ref();
                let fj = self.population[j].fitness.as_ref();

                if let (Some(fi), Some(fj)) = (fi, fj) {
                    match self.dominates(fi, fj) {
                        Some(true) => {
                            // i dominates j
                            dominated_set[i].push(j);
                            domination_count[j] += 1;
                        }
                        Some(false) => {
                            // j dominates i
                            dominated_set[j].push(i);
                            domination_count[i] += 1;
                        }
                        None => {
                            // Neither dominates (non-dominated)
                        }
                    }
                }
            }
        }

        // Assign Pareto ranks using fronts
        let mut current_front: Vec<usize> = (0..n)
            .filter(|&i| domination_count[i] == 0)
            .collect();
        let mut rank = 0;

        while !current_front.is_empty() {
            // Assign rank to current front
            for &i in &current_front {
                if let Some(ref mut fitness) = self.population[i].fitness {
                    fitness.pareto_rank = rank;
                }
            }

            // Compute crowding distance for this front
            self.compute_crowding_distance(&current_front);

            // Build next front
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_set[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }

            current_front = next_front;
            rank += 1;
        }
    }

    /// Check if fitness a dominates fitness b
    /// Returns Some(true) if a dominates b, Some(false) if b dominates a, None if neither
    fn dominates(&self, a: &FitnessScore, b: &FitnessScore) -> Option<bool> {
        let a_values = [a.accuracy, a.efficiency, a.proof_cost];
        let b_values = [b.accuracy, b.efficiency, b.proof_cost];

        let mut a_better_count = 0;
        let mut b_better_count = 0;

        for (av, bv) in a_values.iter().zip(b_values.iter()) {
            if av > bv {
                a_better_count += 1;
            } else if bv > av {
                b_better_count += 1;
            }
        }

        if a_better_count > 0 && b_better_count == 0 {
            Some(true) // a dominates b
        } else if b_better_count > 0 && a_better_count == 0 {
            Some(false) // b dominates a
        } else {
            None // Neither dominates
        }
    }

    /// Compute crowding distance for a Pareto front
    fn compute_crowding_distance(&mut self, front: &[usize]) {
        let n = front.len();
        if n <= 2 {
            // Boundary solutions get infinite distance
            for &i in front {
                if let Some(ref mut fitness) = self.population[i].fitness {
                    fitness.crowding_distance = f64::INFINITY;
                }
            }
            return;
        }

        // Initialize distances to 0
        for &i in front {
            if let Some(ref mut fitness) = self.population[i].fitness {
                fitness.crowding_distance = 0.0;
            }
        }

        // For each objective, sort and compute distances
        for obj_idx in 0..3 {
            // Sort front by this objective
            let mut sorted_front: Vec<usize> = front.to_vec();
            sorted_front.sort_by(|&a, &b| {
                let fa = self.population[a].fitness.as_ref().unwrap();
                let fb = self.population[b].fitness.as_ref().unwrap();
                let va = match obj_idx {
                    0 => fa.accuracy,
                    1 => fa.efficiency,
                    _ => fa.proof_cost,
                };
                let vb = match obj_idx {
                    0 => fb.accuracy,
                    1 => fb.efficiency,
                    _ => fb.proof_cost,
                };
                va.partial_cmp(&vb).unwrap_or(Ordering::Equal)
            });

            // Boundary solutions get infinite distance
            if let Some(ref mut fitness) = self.population[sorted_front[0]].fitness {
                fitness.crowding_distance = f64::INFINITY;
            }
            if let Some(ref mut fitness) = self.population[sorted_front[n - 1]].fitness {
                fitness.crowding_distance = f64::INFINITY;
            }

            // Get objective range
            let f_min = {
                let f = self.population[sorted_front[0]].fitness.as_ref().unwrap();
                match obj_idx {
                    0 => f.accuracy,
                    1 => f.efficiency,
                    _ => f.proof_cost,
                }
            };
            let f_max = {
                let f = self.population[sorted_front[n - 1]].fitness.as_ref().unwrap();
                match obj_idx {
                    0 => f.accuracy,
                    1 => f.efficiency,
                    _ => f.proof_cost,
                }
            };
            let f_range = f_max - f_min;

            if f_range.abs() < 1e-10 {
                continue; // No variation in this objective
            }

            // Compute distances for interior solutions
            for i in 1..(n - 1) {
                let prev_idx = sorted_front[i - 1];
                let next_idx = sorted_front[i + 1];
                let curr_idx = sorted_front[i];

                let f_prev = {
                    let f = self.population[prev_idx].fitness.as_ref().unwrap();
                    match obj_idx {
                        0 => f.accuracy,
                        1 => f.efficiency,
                        _ => f.proof_cost,
                    }
                };
                let f_next = {
                    let f = self.population[next_idx].fitness.as_ref().unwrap();
                    match obj_idx {
                        0 => f.accuracy,
                        1 => f.efficiency,
                        _ => f.proof_cost,
                    }
                };

                if let Some(ref mut fitness) = self.population[curr_idx].fitness {
                    if fitness.crowding_distance.is_finite() {
                        fitness.crowding_distance += (f_next - f_prev) / f_range;
                    }
                }
            }
        }
    }

    /// Crossover two architectures with dimension-aware repair
    fn crossover<R: Rng>(
        &mut self,
        parent1: &Architecture,
        parent2: &Architecture,
        rng: &mut R,
    ) -> Architecture {
        self.stats.total_crossovers += 1;

        // Single-point crossover
        let min_len = parent1.layers.len().min(parent2.layers.len());
        let crossover_point = if min_len > 1 {
            rng.gen_range(1..min_len)
        } else {
            1
        };

        let mut child_layers = Vec::new();

        // Take first part from parent1
        for i in 0..crossover_point {
            child_layers.push(parent1.layers[i].clone());
        }

        // Get the output dimension at crossover point from parent1
        let crossover_dim = child_layers
            .iter()
            .rev()
            .find(|l| l.layer_type == LayerType::Dense)
            .map(|l| l.output_dim)
            .unwrap_or(parent1.input_dim);

        // Take second part from parent2 with dimension repair
        for i in crossover_point..parent2.layers.len() {
            let mut layer = parent2.layers[i].clone();

            // DIMENSION REPAIR: Adapt the first Dense layer after crossover point
            // to match the dimension from parent1's section
            if i == crossover_point && layer.layer_type == LayerType::Dense {
                // This layer's input must match crossover_dim
                // We keep the layer but may need to adjust its position in data flow
                // For non-Dense layers, they preserve dimension, so no issue
            }

            child_layers.push(layer);
        }

        // CRITICAL FIX: Repair dimension chain through the architecture
        self.repair_dimensions(&mut child_layers, parent1.input_dim, parent1.output_dim);

        // Ensure output dimension matches
        if let Some(last) = child_layers.last_mut() {
            if last.layer_type == LayerType::Dense {
                last.output_dim = parent1.output_dim;
            }
        }

        let id = self.next_id;
        self.next_id += 1;

        Architecture {
            id,
            layers: child_layers,
            input_dim: parent1.input_dim,
            output_dim: parent1.output_dim,
            fitness: None,
            generation: self.generation + 1,
            parents: vec![parent1.id, parent2.id],
        }
    }

    /// Repair dimension chain to ensure valid architecture
    fn repair_dimensions(&self, layers: &mut Vec<LayerGene>, input_dim: usize, output_dim: usize) {
        let mut current_dim = input_dim;

        for (i, layer) in layers.iter_mut().enumerate() {
            let is_last = i == layers.len() - 1;

            match layer.layer_type {
                LayerType::Dense => {
                    // Dense layer changes dimension
                    if is_last {
                        layer.output_dim = output_dim;
                    } else {
                        // Ensure output_dim is within bounds
                        layer.output_dim = layer.output_dim
                            .max(self.config.min_width)
                            .min(self.config.max_width);
                    }
                    current_dim = layer.output_dim;
                }
                LayerType::ReLU | LayerType::Sigmoid | LayerType::Tanh |
                LayerType::BatchNorm | LayerType::Softmax => {
                    // These layers preserve dimension
                    layer.output_dim = current_dim;
                }
                LayerType::Dropout => {
                    // Dropout preserves dimension
                    layer.output_dim = current_dim;
                }
                LayerType::ResidualConnection => {
                    // Residual connection must match skip target dimension
                    layer.output_dim = current_dim;
                }
            }
        }

        // If no Dense layer at the end, add one
        if layers.last().map(|l| l.layer_type != LayerType::Dense).unwrap_or(true) {
            layers.push(LayerGene {
                layer_type: LayerType::Dense,
                output_dim,
                activation_strength: 0.5,
                dropout_rate: 0.0,
                skip_target: None,
            });
        }
    }

    /// Mutate architecture
    fn mutate<R: Rng>(&self, arch: &mut Architecture, rng: &mut R) {
        let mutation_type = rng.gen_range(0..5);

        match mutation_type {
            0 => {
                // Add layer
                if arch.layers.len() < self.config.max_depth {
                    let insert_pos = rng.gen_range(0..arch.layers.len());
                    let layer_type = *self.config.layer_types.choose(rng).unwrap_or(&LayerType::ReLU);
                    let prev_dim = if insert_pos > 0 {
                        arch.layers[insert_pos - 1].output_dim
                    } else {
                        arch.input_dim
                    };

                    let gene = LayerGene {
                        layer_type,
                        output_dim: if layer_type == LayerType::Dense {
                            rng.gen_range(self.config.min_width..=self.config.max_width)
                        } else {
                            prev_dim
                        },
                        activation_strength: rng.gen(),
                        dropout_rate: if layer_type == LayerType::Dropout {
                            rng.gen_range(0.1..0.5)
                        } else {
                            0.0
                        },
                        skip_target: None,
                    };

                    arch.layers.insert(insert_pos, gene);
                    debug!("Mutation: Added layer at position {}", insert_pos);
                }
            }
            1 => {
                // Remove layer (if not too small)
                if arch.layers.len() > 2 {
                    let remove_pos = rng.gen_range(0..arch.layers.len() - 1); // Don't remove last
                    arch.layers.remove(remove_pos);
                    debug!("Mutation: Removed layer at position {}", remove_pos);
                }
            }
            2 => {
                // Modify layer width
                let layer_idx = rng.gen_range(0..arch.layers.len());
                if arch.layers[layer_idx].layer_type == LayerType::Dense {
                    let old_dim = arch.layers[layer_idx].output_dim;
                    let delta: i32 = rng.gen_range(-64..=64);
                    let new_dim = (old_dim as i32 + delta)
                        .max(self.config.min_width as i32)
                        .min(self.config.max_width as i32) as usize;
                    arch.layers[layer_idx].output_dim = new_dim;
                    debug!("Mutation: Changed width {} -> {}", old_dim, new_dim);
                }
            }
            3 => {
                // Change layer type
                let layer_idx = rng.gen_range(0..arch.layers.len().saturating_sub(1));
                let old_type = arch.layers[layer_idx].layer_type;
                let new_type = *self.config.layer_types.choose(rng).unwrap_or(&LayerType::ReLU);
                arch.layers[layer_idx].layer_type = new_type;
                debug!("Mutation: Changed layer type {:?} -> {:?}", old_type, new_type);
            }
            4 => {
                // Modify activation strength
                let layer_idx = rng.gen_range(0..arch.layers.len());
                let delta: f64 = rng.gen_range(-0.2..=0.2);
                arch.layers[layer_idx].activation_strength =
                    (arch.layers[layer_idx].activation_strength + delta).clamp(0.0, 1.0);
            }
            _ => {}
        }

        arch.fitness = None; // Invalidate fitness after mutation
    }

    /// Update hall of fame with best unique architectures
    fn update_hall_of_fame(&mut self) {
        let max_hof_size = 10;

        for arch in self.population.iter().take(3) {
            // Check if similar architecture already in HoF
            let dominated = self.hall_of_fame.iter().any(|hof| {
                hof.fitness.as_ref().unwrap().combined >= arch.fitness.as_ref().unwrap().combined
            });

            if !dominated && self.hall_of_fame.len() < max_hof_size {
                self.hall_of_fame.push(arch.clone());
            }
        }

        // Sort and trim
        self.hall_of_fame.sort_by(|a, b| {
            b.fitness
                .as_ref()
                .unwrap()
                .combined
                .partial_cmp(&a.fitness.as_ref().unwrap().combined)
                .unwrap_or(Ordering::Equal)
        });
        self.hall_of_fame.truncate(max_hof_size);
    }

    /// Get best architecture found
    pub fn best(&self) -> Option<&Architecture> {
        self.best.as_ref()
    }

    /// Get hall of fame
    pub fn hall_of_fame(&self) -> &[Architecture] {
        &self.hall_of_fame
    }

    /// Get current population
    pub fn population(&self) -> &[Architecture] {
        &self.population
    }

    /// Get statistics
    pub fn stats(&self) -> &NasStats {
        &self.stats
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Run full evolution
    pub fn run<F>(&mut self, input_dim: usize, output_dim: usize, evaluator: F) -> Result<Architecture, NasError>
    where
        F: Fn(&Architecture) -> FitnessScore,
    {
        self.initialize(input_dim, output_dim);

        for gen in 0..self.config.max_generations {
            self.evolve(&evaluator)?;

            // Early stopping if converged
            if gen > 10 {
                let recent_best: Vec<_> = self.stats.best_fitness_history.iter()
                    .rev()
                    .take(10)
                    .collect();

                if recent_best.len() == 10 {
                    let variance: f64 = {
                        let mean = recent_best.iter().copied().sum::<f64>() / 10.0;
                        recent_best.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 10.0
                    };

                    if variance < 1e-6 {
                        info!("🏁 Early stopping: fitness converged at generation {}", gen);
                        break;
                    }
                }
            }
        }

        self.best.clone().ok_or(NasError::NoSolutionFound)
    }
}

/// Quantum-inspired mutation operator
pub struct QuantumMutationOperator {
    /// Superposition probability
    superposition_prob: f64,

    /// Entanglement strength
    entanglement_strength: f64,
}

impl QuantumMutationOperator {
    /// Create new quantum mutation operator
    pub fn new(superposition_prob: f64, entanglement_strength: f64) -> Self {
        Self {
            superposition_prob,
            entanglement_strength,
        }
    }

    /// Apply quantum-inspired mutation
    pub fn mutate(&self, arch: &mut Architecture, rng: &mut impl Rng) {
        // Superposition: consider multiple mutations simultaneously
        if rng.gen::<f64>() < self.superposition_prob {
            // Apply multiple small mutations
            let num_mutations = rng.gen_range(2..=4);
            for _ in 0..num_mutations {
                self.apply_small_mutation(arch, rng);
            }
        }

        // Entanglement: correlated mutations across layers
        if rng.gen::<f64>() < self.entanglement_strength {
            self.apply_entangled_mutation(arch, rng);
        }
    }

    fn apply_small_mutation(&self, arch: &mut Architecture, rng: &mut impl Rng) {
        if arch.layers.is_empty() {
            return;
        }

        let idx = rng.gen_range(0..arch.layers.len());
        arch.layers[idx].activation_strength = rng.gen();
    }

    fn apply_entangled_mutation(&self, arch: &mut Architecture, rng: &mut impl Rng) {
        // Scale all Dense layer widths by same factor
        let scale = rng.gen_range(0.8..=1.2);
        for layer in &mut arch.layers {
            if layer.layer_type == LayerType::Dense {
                layer.output_dim = ((layer.output_dim as f64 * scale) as usize).max(8);
            }
        }
    }
}

/// NAS errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum NasError {
    #[error("No solution found after evolution")]
    NoSolutionFound,

    #[error("Population not initialized")]
    NotInitialized,

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),
}

/// Simple fitness evaluator for testing
pub fn simple_evaluator(arch: &Architecture) -> FitnessScore {
    // Prefer: moderate depth, reasonable width, good layer mix
    let depth_score = 1.0 / (1.0 + ((arch.layers.len() as f64 - 5.0) / 3.0).abs());

    let avg_width = arch
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Dense)
        .map(|l| l.output_dim)
        .sum::<usize>() as f64
        / arch.layers.iter().filter(|l| l.layer_type == LayerType::Dense).count().max(1) as f64;
    let width_score = 1.0 / (1.0 + ((avg_width - 128.0) / 64.0).abs());

    let param_count: usize = arch
        .layers
        .iter()
        .filter(|l| l.layer_type == LayerType::Dense)
        .map(|l| l.output_dim * l.output_dim)
        .sum();
    let efficiency = 1.0 / (1.0 + (param_count as f64 / 10000.0).ln());

    let accuracy = depth_score * 0.5 + width_score * 0.5;
    let proof_cost = 1.0 / (arch.layers.len() as f64);

    let combined = 0.7 * accuracy + 0.2 * efficiency + 0.1 * proof_cost;

    FitnessScore {
        accuracy,
        efficiency,
        proof_cost,
        combined,
        pareto_rank: 0,
        crowding_distance: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_initialization() {
        let config = NasConfig {
            population_size: 10,
            max_generations: 5,
            ..Default::default()
        };

        let mut engine = NasEngine::new(config);
        engine.initialize(64, 10);

        assert_eq!(engine.population().len(), 10);
        for arch in engine.population() {
            assert_eq!(arch.input_dim, 64);
            assert_eq!(arch.output_dim, 10);
            assert!(!arch.layers.is_empty());
        }
    }

    #[test]
    fn test_nas_evolution() {
        let config = NasConfig {
            population_size: 10,
            max_generations: 3,
            ..Default::default()
        };

        let mut engine = NasEngine::new(config);
        engine.initialize(64, 10);

        for _ in 0..3 {
            engine.evolve(simple_evaluator).unwrap();
        }

        assert!(engine.best().is_some());
        assert!(engine.stats().total_evaluations > 0);
    }

    #[test]
    fn test_nas_full_run() {
        let config = NasConfig {
            population_size: 10,
            max_generations: 5,
            ..Default::default()
        };

        let mut engine = NasEngine::new(config);
        let best = engine.run(32, 8, simple_evaluator).unwrap();

        assert!(best.fitness.is_some());
        assert!(best.fitness.as_ref().unwrap().combined > 0.0);
    }

    #[test]
    fn test_quantum_mutation() {
        let mut arch = Architecture {
            id: 0,
            layers: vec![
                LayerGene {
                    layer_type: LayerType::Dense,
                    output_dim: 64,
                    activation_strength: 0.5,
                    dropout_rate: 0.0,
                    skip_target: None,
                },
                LayerGene {
                    layer_type: LayerType::ReLU,
                    output_dim: 64,
                    activation_strength: 0.5,
                    dropout_rate: 0.0,
                    skip_target: None,
                },
            ],
            input_dim: 32,
            output_dim: 64,
            fitness: None,
            generation: 0,
            parents: vec![],
        };

        let mut rng = rand::thread_rng();
        let operator = QuantumMutationOperator::new(0.5, 0.3);
        operator.mutate(&mut arch, &mut rng);

        // Architecture should still be valid after mutation
        assert!(!arch.layers.is_empty());
    }
}
