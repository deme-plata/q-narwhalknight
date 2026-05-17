/// High-Performance WebSocket Binary Protocol Benchmark
/// Tests the actual parallel workers capacity with Rust client

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use rmp_serde as rmps;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::Semaphore;

const SERVER_URL: &str = "ws://localhost:9050/api/v1/binary/stream";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    id: Vec<u8>,
    from: Vec<u8>,
    to: Vec<u8>,
    amount: u64,
    fee: u64,
    nonce: u64,
    signature: Vec<u8>,
    timestamp: String,
    data: Vec<u8>,
}

fn create_address(seed: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    hasher.finalize().to_vec()
}

fn create_signature(data: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let hash1 = hasher.finalize();

    let mut hasher = Sha256::new();
    hasher.update(&hash1);
    let hash2 = hasher.finalize();

    [&hash1[..], &hash2[..]].concat()
}

fn create_test_transaction(index: usize) -> Transaction {
    let from_addr = create_address(&format!("from_{}", index));
    let to_addr = create_address(&format!("to_{}", index));
    let tx_id = create_address(&format!("tx_{}_{}", index, index));
    let signature = create_signature(&format!("sign_{}", index));

    Transaction {
        id: tx_id,
        from: from_addr,
        to: to_addr,
        amount: 1000 + index as u64,
        fee: 10,
        nonce: index as u64,
        signature,
        timestamp: chrono::Utc::now().to_rfc3339(),
        data: vec![],
    }
}

async fn benchmark_websocket_stream(num_transactions: usize) -> f64 {
    let url = url::Url::parse(SERVER_URL).expect("Invalid URL");

    let (ws_stream, _) = connect_async(url)
        .await
        .expect("Failed to connect");

    let (mut write, _read) = ws_stream.split();

    let start = Instant::now();

    for i in 0..num_transactions {
        let tx = create_test_transaction(i);
        let packed = rmps::to_vec(&tx).expect("Failed to pack");
        write.send(Message::Binary(packed)).await.expect("Failed to send");
    }

    let elapsed = start.elapsed();
    num_transactions as f64 / elapsed.as_secs_f64()
}

async fn benchmark_parallel_clients(num_clients: usize, txs_per_client: usize) -> (f64, Vec<f64>) {
    let semaphore = Arc::new(Semaphore::new(num_clients));
    let mut tasks = Vec::new();

    let start = Instant::now();

    for client_id in 0..num_clients {
        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            let url = url::Url::parse(SERVER_URL).expect("Invalid URL");
            let (ws_stream, _) = connect_async(url).await.expect("Failed to connect");
            let (mut write, _read) = ws_stream.split();

            let client_start = Instant::now();

            for i in 0..txs_per_client {
                let tx = create_test_transaction(client_id * txs_per_client + i);
                let packed = rmps::to_vec(&tx).expect("Failed to pack");
                write.send(Message::Binary(packed)).await.expect("Failed to send");
            }

            let elapsed = client_start.elapsed();
            txs_per_client as f64 / elapsed.as_secs_f64()
        });

        tasks.push(task);
    }

    let mut client_tps = Vec::new();
    for task in tasks {
        if let Ok(tps) = task.await {
            client_tps.push(tps);
        }
    }

    let total_elapsed = start.elapsed();
    let total_txs = (num_clients * txs_per_client) as f64;
    let aggregate_tps = total_txs / total_elapsed.as_secs_f64();

    (aggregate_tps, client_tps)
}

fn benchmark_single_client(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("websocket_binary_single_client");

    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                black_box(benchmark_websocket_stream(size).await)
            });
        });
    }

    group.finish();
}

fn benchmark_parallel_clients_test(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("websocket_binary_parallel");

    // Test configurations: (clients, txs_per_client)
    for (clients, txs) in [(4, 1000), (8, 1000), (16, 1000)].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}clients_{}tx", clients, txs)),
            &(*clients, *txs),
            |b, &(clients, txs)| {
                b.to_async(&rt).iter(|| async move {
                    black_box(benchmark_parallel_clients(clients, txs).await)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_single_client, benchmark_parallel_clients_test);
criterion_main!(benches);
