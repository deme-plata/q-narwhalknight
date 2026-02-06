//! Consensus Performance Benchmarking Module
//!
//! Measures actual DAG vertex creation, validation, and finality timing
//! for the Q-NarwhalKnight consensus system.

use crate::{BenchmarkConfig, ConsensusMetrics};
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

use q_types::{
    Address, Amount, NodeId, Round, SecretKey, Sha3_256, Digest, TokenType, Transaction,
    TransactionType, TxHash, TxSignaturePhase, Vertex, VertexId,
};

/// Create a realistic test transaction with actual Ed25519 signing
fn create_signed_transaction(signing_key: &SecretKey, nonce: u64) -> Transaction {
    use ed25519_dalek::Signer;

    let from: Address = signing_key.verifying_key().to_bytes();
    let to: Address = [0xBBu8; 32];

    // Build the canonical message to sign (from || to || amount || nonce)
    let amount: Amount = 1_000_000;
    let fee: Amount = 100;
    let mut msg = Vec::with_capacity(32 + 32 + 16 + 16 + 8);
    msg.extend_from_slice(&from);
    msg.extend_from_slice(&to);
    msg.extend_from_slice(&amount.to_le_bytes());
    msg.extend_from_slice(&fee.to_le_bytes());
    msg.extend_from_slice(&nonce.to_le_bytes());

    let signature = signing_key.sign(&msg);

    Transaction {
        id: {
            let mut hasher = Sha3_256::new();
            hasher.update(&msg);
            let result = hasher.finalize();
            let mut id = [0u8; 32];
            id.copy_from_slice(&result);
            id
        },
        from,
        to,
        amount,
        fee,
        nonce,
        signature: signature.to_bytes().to_vec(),
        timestamp: chrono::Utc::now(),
        data: Vec::new(),
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: Default::default(),
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

/// Create a DAG vertex containing a batch of transactions
fn create_vertex(
    round: Round,
    author: NodeId,
    transactions: Vec<Transaction>,
    parents: Vec<VertexId>,
) -> Vertex {
    // Compute tx_root as hash of all transaction ids
    let mut hasher = Sha3_256::new();
    for tx in &transactions {
        hasher.update(tx.id);
    }
    let root = hasher.finalize();
    let mut tx_root: TxHash = [0u8; 32];
    tx_root.copy_from_slice(&root);

    // Compute vertex id
    let mut id_hasher = Sha3_256::new();
    id_hasher.update(round.to_le_bytes());
    id_hasher.update(author);
    id_hasher.update(tx_root);
    for p in &parents {
        id_hasher.update(p);
    }
    let id_hash = id_hasher.finalize();
    let mut id: VertexId = [0u8; 32];
    id.copy_from_slice(&id_hash);

    Vertex {
        id,
        round,
        author,
        tx_root,
        parents,
        transactions,
        signature: vec![0u8; 64], // placeholder sig for benchmarking vertex creation speed
        timestamp: chrono::Utc::now(),
    }
}

/// Validate a vertex by checking its tx_root hash
fn validate_vertex(vertex: &Vertex) -> bool {
    let mut hasher = Sha3_256::new();
    for tx in &vertex.transactions {
        hasher.update(tx.id);
    }
    let root = hasher.finalize();
    let mut expected_root: TxHash = [0u8; 32];
    expected_root.copy_from_slice(&root);
    expected_root == vertex.tx_root
}

/// Traverse the DAG from a vertex back through its parents to compute finality depth.
/// Returns the number of ancestor layers visited.
fn traverse_dag_depth(
    vertex_id: &VertexId,
    dag: &HashMap<VertexId, Vertex>,
    max_depth: usize,
) -> usize {
    let mut current_layer = vec![*vertex_id];
    let mut depth = 0;

    while depth < max_depth && !current_layer.is_empty() {
        let mut next_layer = Vec::new();
        for vid in &current_layer {
            if let Some(v) = dag.get(vid) {
                next_layer.extend_from_slice(&v.parents);
            }
        }
        if next_layer.is_empty() {
            break;
        }
        current_layer = next_layer;
        depth += 1;
    }
    depth
}

pub async fn measure_consensus_performance(config: &BenchmarkConfig) -> Result<ConsensusMetrics> {
    info!("Measuring consensus performance with actual vertex operations");

    let num_validators = config.validator_count.max(1) as usize;
    let rounds_to_simulate = 50; // enough rounds to get stable measurements
    let txs_per_vertex = 100; // realistic batch size

    // Generate signing keys for each validator
    let signing_keys: Vec<SecretKey> = (0..num_validators)
        .map(|i| {
            let mut seed = [0u8; 32];
            seed[0] = i as u8;
            seed[1] = 0xAA;
            SecretKey::from_bytes(&seed)
        })
        .collect();

    let validator_ids: Vec<NodeId> = signing_keys
        .iter()
        .map(|sk| sk.verifying_key().to_bytes())
        .collect();

    // =========================================================================
    // Phase 1: Measure vertex creation rate (create vertices with real hashing)
    // =========================================================================
    let mut dag: HashMap<VertexId, Vertex> = HashMap::new();
    let mut latest_vertex_ids: Vec<VertexId> = Vec::new();
    let mut vertex_creation_times: Vec<f64> = Vec::new();
    let mut validation_times: Vec<f64> = Vec::new();
    let mut round_latencies: Vec<f64> = Vec::new();
    let mut total_vertices_created: u64 = 0;

    let overall_start = Instant::now();

    for round in 0..rounds_to_simulate as u64 {
        let round_start = Instant::now();

        let parents = latest_vertex_ids.clone();
        let mut new_ids = Vec::new();

        for v_idx in 0..num_validators {
            // Create signed transactions for this vertex
            let tx_create_start = Instant::now();
            let transactions: Vec<Transaction> = (0..txs_per_vertex)
                .map(|n| {
                    create_signed_transaction(
                        &signing_keys[v_idx],
                        round * txs_per_vertex as u64 + n as u64,
                    )
                })
                .collect();

            // Create the vertex
            let vertex = create_vertex(
                round,
                validator_ids[v_idx],
                transactions,
                parents.clone(),
            );
            let creation_elapsed = tx_create_start.elapsed();
            vertex_creation_times.push(creation_elapsed.as_secs_f64() * 1000.0);

            // Validate the vertex
            let val_start = Instant::now();
            let valid = validate_vertex(&vertex);
            let val_elapsed = val_start.elapsed();
            validation_times.push(val_elapsed.as_secs_f64() * 1000.0);

            assert!(valid, "Vertex validation should pass for correctly created vertex");

            new_ids.push(vertex.id);
            dag.insert(vertex.id, vertex);
            total_vertices_created += 1;
        }

        latest_vertex_ids = new_ids;

        let round_elapsed = round_start.elapsed();
        round_latencies.push(round_elapsed.as_secs_f64() * 1000.0);
    }

    let overall_elapsed = overall_start.elapsed();

    // =========================================================================
    // Phase 2: Measure DAG traversal time (finality calculation)
    // =========================================================================
    let traversal_start = Instant::now();
    let mut traversal_depths = Vec::new();
    for vid in &latest_vertex_ids {
        let depth = traverse_dag_depth(vid, &dag, rounds_to_simulate);
        traversal_depths.push(depth);
    }
    let traversal_elapsed = traversal_start.elapsed();

    // =========================================================================
    // Compute final metrics from real measurements
    // =========================================================================
    let vertex_processing_rate =
        total_vertices_created as f64 / overall_elapsed.as_secs_f64();

    let dag_growth_rate = dag.len() as f64 / overall_elapsed.as_secs_f64();

    // Finality latency: average time for one full round (all validators produce a vertex)
    let mean_round_latency =
        round_latencies.iter().sum::<f64>() / round_latencies.len() as f64;

    // Consensus efficiency: fraction of validators that participated (always 1.0 in simulation)
    // Weighted by how close vertex creation is to the theoretical minimum time
    let mean_creation_ms =
        vertex_creation_times.iter().sum::<f64>() / vertex_creation_times.len() as f64;
    let mean_validation_ms =
        validation_times.iter().sum::<f64>() / validation_times.len() as f64;

    // Efficiency: what fraction of round time is actual work vs overhead
    let useful_work_per_round_ms =
        (mean_creation_ms + mean_validation_ms) * num_validators as f64;
    let consensus_efficiency = if mean_round_latency > 0.0 {
        (useful_work_per_round_ms / mean_round_latency).min(1.0)
    } else {
        1.0
    };

    info!("Consensus benchmark results:");
    info!(
        "  Vertices created: {} in {:.2}s",
        total_vertices_created,
        overall_elapsed.as_secs_f64()
    );
    info!("  Vertex processing rate: {:.0}/s", vertex_processing_rate);
    info!("  DAG growth rate: {:.0} vertices/s", dag_growth_rate);
    info!("  Mean round latency: {:.2}ms", mean_round_latency);
    info!("  Mean vertex creation: {:.3}ms", mean_creation_ms);
    info!("  Mean vertex validation: {:.4}ms", mean_validation_ms);
    info!(
        "  DAG traversal ({} tips): {:.2}ms",
        latest_vertex_ids.len(),
        traversal_elapsed.as_secs_f64() * 1000.0
    );
    info!("  Consensus efficiency: {:.1}%", consensus_efficiency * 100.0);

    Ok(ConsensusMetrics {
        vertex_processing_rate,
        dag_growth_rate,
        finality_latency_ms: mean_round_latency,
        consensus_efficiency,
        validator_participation: 1.0, // all simulated validators always participate
    })
}
