//! 🧅 Tor stem RECEIVER — the accept side that completes the Dandelion stem relay.
//!
//! Sender half (`QTorClient::send_over_tor`) connects via SOCKS5 to a peer's onion address
//! and writes a length-framed message (`[u32 BE len][bytes]`). This module is the matching
//! receiver: it binds the local port the node's onion service forwards to, reads those
//! frames, and hands each payload to the node over an mpsc channel. The node then injects
//! the tx into its mempool and fluffs it into the open gossip mesh — so the stem relay
//! completes end-to-end:
//!
//!   Node A: mixer tx → send_over_tor ──(SOCKS5 → Tor circuit → onion)──▶ Node B onion svc
//!                                                                         → stem_receiver
//!                                                                         → mempool + fluff
//!
//! The payload is the same `postcard`-serialized `q_types::Transaction` the gossip path
//! carries, so the node deserializes it with the existing tx-ingest logic.

use anyhow::{anyhow, Result};
use std::net::SocketAddr;
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// 8 MiB cap — a transaction is tiny; this is just a sanity bound against a hostile length.
const MAX_STEM_MSG: usize = 8 * 1024 * 1024;

/// Read one length-framed message (`[u32 BE len][bytes]`) — the exact framing
/// `QTorClient::send_over_tor` writes.
pub async fn read_framed(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 || len > MAX_STEM_MSG {
        return Err(anyhow!("invalid stem frame length {len}"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Bind `bind_addr` and forward every received stem payload to `tx`. One message per
/// connection (stem relays are one-shot, matching the sender). Runs until the listener
/// errors or the consumer is dropped.
pub async fn serve(bind_addr: SocketAddr, tx: mpsc::Sender<Vec<u8>>) -> Result<()> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| anyhow!("stem receiver bind {bind_addr}: {e}"))?;
    info!("🧅 [STEM-RX] listening on {} for Tor stem relays", bind_addr);
    loop {
        match listener.accept().await {
            Ok((mut stream, peer)) => {
                let tx = tx.clone();
                tokio::spawn(async move {
                    match read_framed(&mut stream).await {
                        Ok(msg) => {
                            debug!("🧅 [STEM-RX] received {} bytes from {}", msg.len(), peer);
                            if tx.send(msg).await.is_err() {
                                warn!("🧅 [STEM-RX] consumer dropped; stopping forward");
                            }
                        }
                        Err(e) => debug!("🧅 [STEM-RX] read from {} failed: {}", peer, e),
                    }
                });
            }
            Err(e) => {
                warn!("🧅 [STEM-RX] accept error: {}", e);
                return Err(anyhow!("stem receiver accept: {e}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn stem_receiver_reads_framed_message() {
        let (tx, mut rx) = mpsc::channel(4);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Minimal serve: accept one connection, read one frame, forward it.
        tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let msg = read_framed(&mut s).await.unwrap();
            tx.send(msg).await.unwrap();
        });

        // Client writes a frame with the SAME framing as send_over_tor.
        let payload = b"postcard-serialized-transaction-bytes";
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(&(payload.len() as u32).to_be_bytes()).await.unwrap();
        c.write_all(payload).await.unwrap();
        c.flush().await.unwrap();

        let got = rx.recv().await.unwrap();
        assert_eq!(got, payload, "receiver must recover the exact stem payload");
    }

    #[tokio::test]
    async fn rejects_oversized_frame_header() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let h = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            read_framed(&mut s).await
        });
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
        c.flush().await.unwrap();
        assert!(h.await.unwrap().is_err(), "oversized length must be rejected");
    }
}
