//! Network Performance Benchmarking Module
//!
//! Measures actual serialization throughput, message-passing latency via tokio channels,
//! and calculates realistic network capacity from measured data sizes.

use crate::{BenchmarkConfig, NetworkMetrics};
use anyhow::Result;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, info};

use q_types::{
    Address, Amount, SecretKey, Sha3_256, Digest, TokenType, Transaction,
    TransactionType, TxSignaturePhase,
};

/// Create a realistic transaction for serialization benchmarks
fn create_test_transaction(nonce: u64) -> Transaction {
    use ed25519_dalek::Signer;

    let mut seed = [0u8; 32];
    seed[0] = 0x01;
    let signing_key = SecretKey::from_bytes(&seed);
    let from: Address = signing_key.verifying_key().to_bytes();
    let to: Address = [0xCCu8; 32];
    let amount: Amount = 500_000;
    let fee: Amount = 50;

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

pub async fn measure_network_performance(config: &BenchmarkConfig) -> Result<NetworkMetrics> {
    info!("Measuring network performance with real serialization and channel throughput");

    let batch_size = 100usize;
    let num_batches = 50usize;

    // =========================================================================
    // Phase 1: Measure bincode serialization throughput
    // =========================================================================
    let mut single_tx_serialized_size = 0usize;
    let mut total_serialized_bytes = 0u64;

    let ser_start = Instant::now();
    for batch_idx in 0..num_batches {
        let transactions: Vec<Transaction> = (0..batch_size)
            .map(|i| create_test_transaction((batch_idx * batch_size + i) as u64))
            .collect();

        let encoded = bincode::serialize(&transactions)
            .expect("bincode serialization should not fail");
        total_serialized_bytes += encoded.len() as u64;

        if batch_idx == 0 {
            // Also measure single-tx size for capacity calculation
            let single = bincode::serialize(&transactions[0])
                .expect("single tx serialize should not fail");
            single_tx_serialized_size = single.len();
        }

        // Deserialize to measure round-trip
        let _decoded: Vec<Transaction> = bincode::deserialize(&encoded)
            .expect("bincode deserialization should not fail");
    }
    let ser_elapsed = ser_start.elapsed();

    let serialization_throughput_mbps =
        (total_serialized_bytes as f64 * 8.0) / (ser_elapsed.as_secs_f64() * 1_000_000.0);

    info!(
        "  Serialization throughput: {:.1} Mbps ({} bytes in {:.3}s)",
        serialization_throughput_mbps,
        total_serialized_bytes,
        ser_elapsed.as_secs_f64()
    );
    info!(
        "  Single transaction serialized size: {} bytes",
        single_tx_serialized_size
    );

    // =========================================================================
    // Phase 2: Measure tokio channel message-passing latency (simulates P2P)
    // =========================================================================
    let message_count = 1000usize;
    let (tx_sender, mut rx_receiver) = mpsc::channel::<(Instant, Vec<u8>)>(message_count);

    // Pre-serialize messages
    let test_tx = create_test_transaction(0);
    let serialized_msg = bincode::serialize(&test_tx).expect("serialize");

    // Spawn receiver task
    let receiver_handle = tokio::spawn(async move {
        let mut latencies = Vec::with_capacity(message_count);
        let mut received = 0usize;

        while let Some((send_time, _payload)) = rx_receiver.recv().await {
            let latency = send_time.elapsed();
            latencies.push(latency.as_secs_f64() * 1000.0); // ms
            received += 1;
            if received >= message_count {
                break;
            }
        }
        latencies
    });

    // Send messages and measure
    let channel_start = Instant::now();
    for _ in 0..message_count {
        let msg_clone = serialized_msg.clone();
        tx_sender
            .send((Instant::now(), msg_clone))
            .await
            .expect("channel send should not fail");
    }
    drop(tx_sender);

    let latencies = receiver_handle.await?;
    let channel_elapsed = channel_start.elapsed();

    let mean_message_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };

    let channel_throughput_msgs_per_sec =
        message_count as f64 / channel_elapsed.as_secs_f64();
    let channel_throughput_mbps = (message_count as f64 * serialized_msg.len() as f64 * 8.0)
        / (channel_elapsed.as_secs_f64() * 1_000_000.0);

    info!(
        "  Channel message latency (mean): {:.4}ms",
        mean_message_latency_ms
    );
    info!(
        "  Channel throughput: {:.0} msgs/s ({:.1} Mbps)",
        channel_throughput_msgs_per_sec, channel_throughput_mbps
    );

    // =========================================================================
    // Phase 3: Calculate theoretical network capacity
    // =========================================================================
    // Assume 1 Gbps link between nodes for theoretical capacity
    let assumed_link_speed_mbps = 1000.0_f64;
    let bits_per_tx = (single_tx_serialized_size as f64) * 8.0;
    let theoretical_max_tps = (assumed_link_speed_mbps * 1_000_000.0) / bits_per_tx;
    let bandwidth_utilization = serialization_throughput_mbps / assumed_link_speed_mbps;

    info!(
        "  Theoretical max TPS on 1Gbps link: {:.0}",
        theoretical_max_tps
    );
    info!(
        "  Bandwidth utilization (serialization): {:.1}%",
        bandwidth_utilization * 100.0
    );

    // =========================================================================
    // Phase 4: Simulate packet loss with channel drops
    // =========================================================================
    let loss_test_count = 10_000usize;
    let (loss_tx, mut loss_rx) = mpsc::channel::<u64>(64); // small buffer to induce drops

    let loss_sender = tokio::spawn(async move {
        let mut sent = 0u64;
        for i in 0..loss_test_count as u64 {
            // try_send will fail if buffer is full, simulating packet loss
            if loss_tx.try_send(i).is_ok() {
                sent += 1;
            }
            // Small yield to give receiver time
            if i % 100 == 0 {
                tokio::task::yield_now().await;
            }
        }
        sent
    });

    let loss_receiver = tokio::spawn(async move {
        let mut received = 0u64;
        // Drain the channel - add a small delay to simulate processing
        while loss_rx.recv().await.is_some() {
            received += 1;
            if received % 50 == 0 {
                tokio::task::yield_now().await;
            }
        }
        received
    });

    let (sent, received) = tokio::try_join!(loss_sender, loss_receiver)?;
    let packet_loss_rate = if sent > 0 {
        1.0 - (received as f64 / loss_test_count as f64)
    } else {
        0.0
    };

    info!(
        "  Simulated packet loss: {:.3}% ({}/{} delivered)",
        packet_loss_rate * 100.0,
        received,
        loss_test_count
    );

    // Use the combined throughput: the higher of serialization and channel measurements
    let effective_throughput_mbps = serialization_throughput_mbps.max(channel_throughput_mbps);

    Ok(NetworkMetrics {
        throughput_mbps: effective_throughput_mbps,
        connection_count: config.node_count,
        message_latency_ms: mean_message_latency_ms,
        packet_loss_rate,
        bandwidth_utilization,
    })
}
