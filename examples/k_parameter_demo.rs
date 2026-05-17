//! 🎯 K-Parameter Quantum Phase Analysis Demo
//!
//! This example demonstrates Kristensen's K-Parameter system integrated
//! with Quillon Resonance consensus for quantum-inspired phase transition detection.

use q_resonance::{
    KParameterAnalyzer, KEnhancedEnergy, KParameterMetrics,
    StringState, ResonanceVertex, PhaseTransition,
};
use num_complex::Complex64;
use std::time::Instant;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     🎯 K-PARAMETER QUANTUM PHASE ANALYSIS DEMO 🎯         ║");
    println!("║                                                            ║");
    println!("║  K = 2π √(ΔH · Δs · ℏ) / τ                                ║");
    println!("║                                                            ║");
    println!("║  ΔH: Hamiltonian uncertainty (energy variance)            ║");
    println!("║  Δs: Entropy variance (information entropy)               ║");
    println!("║  ℏ:  Reduced Planck constant (quantum scale)              ║");
    println!("║  τ:  Characteristic timescale (round duration)            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create K-Parameter analyzer
    let analyzer = KParameterAnalyzer::new()
        .with_threshold(1.0); // Phase transition threshold

    println!("✅ K-Parameter Analyzer initialized");
    println!("   - Planck constant (ℏ): 1.0 (normalized)");
    println!("   - Phase transition threshold: 1.0\n");

    // Scenario 1: Stable Consensus
    println!("📊 SCENARIO 1: Stable Consensus\n");
    run_stable_consensus_scenario(&analyzer);

    println!("\n" + &"═".repeat(60) + "\n");

    // Scenario 2: Approaching Phase Transition
    println!("📊 SCENARIO 2: Approaching Phase Transition\n");
    run_approaching_transition_scenario(&analyzer);

    println!("\n" + &"═".repeat(60) + "\n");

    // Scenario 3: Critical Phase Transition
    println!("📊 SCENARIO 3: Critical Phase Transition\n");
    run_critical_transition_scenario(&analyzer);

    println!("\n" + &"═".repeat(60) + "\n");

    // Scenario 4: K-Enhanced Energy Minimization
    println!("📊 SCENARIO 4: K-Enhanced Energy Minimization\n");
    run_k_enhanced_minimization();

    println!("\n" + &"═".repeat(60) + "\n");

    // Scenario 5: Metrics and Monitoring
    println!("📊 SCENARIO 5: Metrics and Monitoring\n");
    run_metrics_monitoring();

    println!("\n✨ K-Parameter demo complete!\n");
}

/// Scenario 1: Stable consensus with low K-Parameter
fn run_stable_consensus_scenario(analyzer: &KParameterAnalyzer) {
    // Low energy variance (stable system)
    let energy_variance = 0.5;

    // Low entropy variance (synchronized nodes)
    let entropy_variance = 0.3;

    // Normal round duration
    let round_duration = 1.0;

    let k = analyzer.compute_k_parameter(
        energy_variance,
        entropy_variance,
        round_duration,
    );

    let tuning = analyzer.adjust_consensus_parameters(k);

    println!("   ΔH (Energy Variance):  {:.4}", energy_variance);
    println!("   Δs (Entropy Variance): {:.4}", entropy_variance);
    println!("   τ (Round Duration):    {:.3}s", round_duration);
    println!("   \n   🎯 K-Parameter: {:.4}", k);
    println!("   \n   Phase State: {} (expected)", PhaseTransition::Stable);
    println!("   \n   Consensus Tuning:");
    println!("   - Learning Rate: {:.3}", tuning.learning_rate);
    println!("   - Max Iterations: {}", tuning.max_iterations);
    println!("   - Spectral Threshold: {:.3}", tuning.spectral_threshold);

    println!("\n   💡 Interpretation: Low K indicates stable consensus.");
    println!("      System can use careful, precise convergence.");
}

/// Scenario 2: Approaching phase transition with medium K-Parameter
fn run_approaching_transition_scenario(analyzer: &KParameterAnalyzer) {
    // Moderate energy variance (some instability)
    let energy_variance = 3.0;

    // Moderate entropy variance (partial desynchronization)
    let entropy_variance = 2.5;

    // Normal round duration
    let round_duration = 1.0;

    let k = analyzer.compute_k_parameter(
        energy_variance,
        entropy_variance,
        round_duration,
    );

    let tuning = analyzer.adjust_consensus_parameters(k);

    println!("   ΔH (Energy Variance):  {:.4}", energy_variance);
    println!("   Δs (Entropy Variance): {:.4}", entropy_variance);
    println!("   τ (Round Duration):    {:.3}s", round_duration);
    println!("   \n   🎯 K-Parameter: {:.4}", k);
    println!("   \n   Phase State: {} (expected)", PhaseTransition::Approaching);
    println!("   \n   Consensus Tuning:");
    println!("   - Learning Rate: {:.3}", tuning.learning_rate);
    println!("   - Max Iterations: {}", tuning.max_iterations);
    println!("   - Spectral Threshold: {:.3}", tuning.spectral_threshold);

    println!("\n   ⚠️  Interpretation: Medium K indicates approaching transition.");
    println!("      System increases monitoring and adjusts parameters.");
}

/// Scenario 3: Critical phase transition with high K-Parameter
fn run_critical_transition_scenario(analyzer: &KParameterAnalyzer) {
    // High energy variance (significant instability)
    let energy_variance = 10.0;

    // High entropy variance (network desynchronization)
    let entropy_variance = 8.0;

    // Normal round duration
    let round_duration = 1.0;

    let k = analyzer.compute_k_parameter(
        energy_variance,
        entropy_variance,
        round_duration,
    );

    let tuning = analyzer.adjust_consensus_parameters(k);

    println!("   ΔH (Energy Variance):  {:.4}", energy_variance);
    println!("   Δs (Entropy Variance): {:.4}", entropy_variance);
    println!("   τ (Round Duration):    {:.3}s", round_duration);
    println!("   \n   🎯 K-Parameter: {:.4}", k);
    println!("   \n   Phase State: {} (expected)", PhaseTransition::Critical);
    println!("   \n   Consensus Tuning:");
    println!("   - Learning Rate: {:.3} (emergency mode)", tuning.learning_rate);
    println!("   - Max Iterations: {} (reduced)", tuning.max_iterations);
    println!("   - Spectral Threshold: {:.3} (aggressive)", tuning.spectral_threshold);

    println!("\n   🚨 Interpretation: High K indicates critical transition!");
    println!("      System activates emergency protocols.");
    println!("      Fast, aggressive convergence to restore stability.");
}

/// Scenario 4: K-Enhanced energy minimization
fn run_k_enhanced_minimization() {
    let mut k_energy = KEnhancedEnergy::new();
    k_energy.start_round();

    // Create test vertices with varying phases
    let mut vertices = vec![
        create_vertex(1.0, 1.0, 0.0),
        create_vertex(1.0, 1.0, std::f64::consts::PI / 4.0),
        create_vertex(1.0, 1.0, std::f64::consts::PI / 2.0),
        create_vertex(1.0, 1.0, 3.0 * std::f64::consts::PI / 4.0),
        create_vertex(1.0, 1.0, std::f64::consts::PI),
    ];

    println!("   Initial vertices: {} nodes", vertices.len());
    println!("   Phase spread: 0 to π radians\n");

    let start = Instant::now();
    let result = k_energy.minimize_with_k_guidance(&mut vertices);
    let elapsed = start.elapsed();

    match result {
        Ok((final_energy, k_value, analysis)) => {
            println!("   ✅ Energy minimization complete in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            println!("   \n   Results:");
            println!("   - Final Energy: {:.6}", final_energy);
            println!("   - K-Parameter: {:.4}", k_value);
            println!("   - Phase State: {}", analysis.phase_transition);
            println!("   - Stability: {:.1}%", analysis.stability * 100.0);
            println!("   - Recommendation: {}", analysis.recommendation);

            println!("\n   💡 K-Parameter guided the convergence:");
            println!("      Learning rate: {:.3}", analysis.tuning_applied.learning_rate);
            println!("      Iterations used: {}", analysis.tuning_applied.max_iterations);
        }
        Err(e) => {
            println!("   ❌ Error: {:?}", e);
        }
    }
}

/// Scenario 5: Metrics and monitoring
fn run_metrics_monitoring() {
    let mut metrics = KParameterMetrics::new();

    println!("   Simulating 10 consensus rounds with varying K-Parameter...\n");

    // Simulate stable rounds
    for i in 1..=5 {
        let k = 0.5 + (i as f64) * 0.1; // Gradually increasing
        let analysis = create_mock_analysis(k, PhaseTransition::Stable);
        metrics.update(&analysis);

        println!("   Round {}: K={:.3} | Phase={} | Stability={:.1}%",
            i, k, analysis.phase_transition, analysis.stability * 100.0);
    }

    // Simulate transition
    for i in 6..=8 {
        let k = 2.0 + (i as f64 - 5.0) * 0.5;
        let analysis = create_mock_analysis(k, PhaseTransition::Approaching);
        metrics.update(&analysis);

        println!("   Round {}: K={:.3} | Phase={} | Stability={:.1}%",
            i, k, analysis.phase_transition, analysis.stability * 100.0);
    }

    // Simulate critical
    for i in 9..=10 {
        let k = 5.0 + (i as f64 - 8.0) * 2.0;
        let analysis = create_mock_analysis(k, PhaseTransition::Critical);
        metrics.update(&analysis);

        println!("   Round {}: K={:.3} | Phase={} | Stability={:.1}%",
            i, k, analysis.phase_transition, analysis.stability * 100.0);
    }

    println!("\n   📊 Final Metrics Summary:\n");
    println!("{}", metrics.summary());

    println!("\n   🎯 K-Parameter Trend: {}", metrics.k_trend_direction());
    println!("   ⚠️  Transition Risk: {:.1}%", metrics.transition_risk * 100.0);
    println!("   ✅ System Healthy: {}", metrics.is_healthy());

    println!("\n   📈 Prometheus Metrics Export:\n");
    let prometheus = metrics.export_prometheus_metrics();
    for line in prometheus.lines().take(10) {
        println!("   {}", line);
    }
    println!("   ... (truncated)");
}

// Helper functions

fn create_vertex(amplitude: f64, frequency: f64, phase: f64) -> ResonanceVertex {
    ResonanceVertex {
        hash: [0u8; 32],
        string_state: StringState {
            amplitude,
            frequency,
            phase: Complex64::new(phase.cos(), phase.sin()),
            coupling_strength: 1.0,
        },
        position: vec![0.0, 0.0, 0.0],
        timestamp: 0,
        round: 0,
    }
}

fn create_mock_analysis(k: f64, phase: PhaseTransition) -> q_resonance::PhaseAnalysis {
    use q_resonance::{PhaseRecommendation, ConsensusTuning};

    q_resonance::PhaseAnalysis {
        k_parameter: k,
        energy_variance: k * 0.5,
        entropy_variance: k * 0.3,
        round_duration: 1.0,
        phase_transition: phase,
        stability: (1.0 / (1.0 + k * 0.2)).max(0.1),
        recommendation: match phase {
            PhaseTransition::Stable => PhaseRecommendation::NormalOperation,
            PhaseTransition::Approaching => PhaseRecommendation::AdjustParameters,
            PhaseTransition::Critical => PhaseRecommendation::EmergencyProtocol,
        },
        tuning_applied: ConsensusTuning {
            learning_rate: 0.1,
            max_iterations: 500,
            spectral_threshold: 0.1,
            convergence_tolerance: 1e-6,
        },
    }
}
