🔥 let’s go — that’s a huge win. With 160.58 KH/s, you’ve removed the #1 blocker. Now lock in the stability gains while the momentum’s hot.

# Next 90 minutes: hardening sprint

1. **Wire HeightState (storm killer)**

* Inject into `AppState`, initialize from `qblock:latest`.
* Replace all hot-path height reads with `height.cached()` / `height.rx.changed().await`.
* On successful block persist, call `height.update(new_height).await`.
* On SIGINT/SIGTERM, call `height.mark_shutdown()` → skip binary search.

2. **Global shutdown broadcast**

* `broadcast::channel::<()>(1)` in `AppState`.
* Every long-running task uses `select!{ _ = sd_rx.recv() => break, ... }`.
* Stop HTTP intake → drain storage actor (≤10s) → WAL sync → exit.

3. **DB timeouts + storage actor**

* Wrap all persists with `timeout(5s)` + **retry x3**.
* Producer only advances after ack; if all retries fail, **pause producer** (degraded mode) and alert.

4. **Remove per-block `flush()`**

* Use WAL sync (`WriteOptions::set_sync(true)`), add background flush every 90s.

5. **Bounded intake**

* Mining submissions: `mpsc::channel(1000)`; return **429** on Full, **503** on Closed.

6. **Systemd hygiene (now)**

```
TimeoutStopSec=15
KillSignal=SIGINT
Restart=on-failure
RestartSec=2
LimitNOFILE=1048576
```

# Quick validation (copy/paste)

* **No storm during stop/start**

  ```
  journalctl -u q-api-server --since "10 min ago" | grep -c "binary search"
  ```
* **Shutdown < 15s**

  ```
  systemctl stop q-api-server && time systemctl start q-api-server
  ```
* **Save latency sane**

  ```
  journalctl -u q-api-server --since "10 min ago" | grep -E "save_block.*t_ms"
  ```
* **Height keeps climbing**

  ```
  watch -n5 'curl -s https://quillon.xyz/api/v1/node/status | jq .data.current_height'
  ```

# 24-hour guardrails (alerts you can add today)

* **Solution queue low**

  * `q_solution_queue_depth < 100 for 1m` → WARN
* **No external flow**

  * submissions/sec == 0 for 10m → WARN
* **Producer stalled**

  * `time_since_last_block > 120s for 2m` → CRITICAL
* **Storage pain**

  * `q_block_save_latency_ms{quantile="0.99"} > 800` for 5m → WARN

# Miner-side nits (keeps hashrate sticky)

* Set **systemd** for miners (you already have it) + `Restart=always`.
* Add **exponential backoff** on submit/challenge refresh (max 5–10s).
* Prefer **keep-alive / HTTP/1.1** or WebSocket long-poll if available.
* Stagger challenge refresh jitter (±500ms) to avoid bursts on restarts.

# Low-risk RocksDB knobs (flip once)

* `set_level_compaction_dynamic_level_bytes(true)`
* `set_max_background_jobs(6)` and `set_max_subcompactions(2)`
* Block-based table: bloom filter (10), `cache_index_and_filter_blocks(true)`
* If you scan by height often, add a **prefix extractor** for height keyspace.

# Tiny test matrix (catch regressions)

* ✅ Graceful stop while blocks are flowing → shutdown ≤ 15s, miners reconnect.
* ✅ Persist timeout injection → 3 retries then **producer paused**, API stays up.
* ✅ Mining queue saturation → 429 surfaced; memory flat.
* ✅ Height cache TTL correctness → cache-hit reads < 1µs, full recompute ≥ TTL.

# What “good” looks like for the next 4+ hours

* Height advances every 2–5s.
* `p99(save_block)` < 500 ms; no executor starvation.
* No “binary search” logs during shutdown.
* Uptime crosses **4h** with zero manual restarts.

# Final note

You’ve proven the diagnosis end-to-end: external miners fixed the systemic stalls, and you captured hard evidence of the shutdown storm. Ship the HeightState wiring + shutdown broadcast now, then the timeouts/actor this afternoon. That combo will turn today’s success into **boring**, durable reliability. 🚀

🔥 let’s go — that’s a huge win. With 160.58 KH/s, you’ve removed the #1 blocker. Now lock in the stability gains while the momentum’s hot.

# Next 90 minutes: hardening sprint

1. **Wire HeightState (storm killer)**

* Inject into `AppState`, initialize from `qblock:latest`.
* Replace all hot-path height reads with `height.cached()` / `height.rx.changed().await`.
* On successful block persist, call `height.update(new_height).await`.
* On SIGINT/SIGTERM, call `height.mark_shutdown()` → skip binary search.

2. **Global shutdown broadcast**

* `broadcast::channel::<()>(1)` in `AppState`.
* Every long-running task uses `select!{ _ = sd_rx.recv() => break, ... }`.
* Stop HTTP intake → drain storage actor (≤10s) → WAL sync → exit.

3. **DB timeouts + storage actor**

* Wrap all persists with `timeout(5s)` + **retry x3**.
* Producer only advances after ack; if all retries fail, **pause producer** (degraded mode) and alert.

4. **Remove per-block `flush()`**

* Use WAL sync (`WriteOptions::set_sync(true)`), add background flush every 90s.

5. **Bounded intake**

* Mining submissions: `mpsc::channel(1000)`; return **429** on Full, **503** on Closed.

6. **Systemd hygiene (now)**

```
TimeoutStopSec=15
KillSignal=SIGINT
Restart=on-failure
RestartSec=2
LimitNOFILE=1048576
```

# Quick validation (copy/paste)

* **No storm during stop/start**

  ```
  journalctl -u q-api-server --since "10 min ago" | grep -c "binary search"
  ```
* **Shutdown < 15s**

  ```
  systemctl stop q-api-server && time systemctl start q-api-server
  ```
* **Save latency sane**

  ```
  journalctl -u q-api-server --since "10 min ago" | grep -E "save_block.*t_ms"
  ```
* **Height keeps climbing**

  ```
  watch -n5 'curl -s https://quillon.xyz/api/v1/node/status | jq .data.current_height'
  ```

# 24-hour guardrails (alerts you can add today)

* **Solution queue low**

  * `q_solution_queue_depth < 100 for 1m` → WARN
* **No external flow**

  * submissions/sec == 0 for 10m → WARN
* **Producer stalled**

  * `time_since_last_block > 120s for 2m` → CRITICAL
* **Storage pain**

  * `q_block_save_latency_ms{quantile="0.99"} > 800` for 5m → WARN

# Miner-side nits (keeps hashrate sticky)

* Set **systemd** for miners (you already have it) + `Restart=always`.
* Add **exponential backoff** on submit/challenge refresh (max 5–10s).
* Prefer **keep-alive / HTTP/1.1** or WebSocket long-poll if available.
* Stagger challenge refresh jitter (±500ms) to avoid bursts on restarts.

# Low-risk RocksDB knobs (flip once)

* `set_level_compaction_dynamic_level_bytes(true)`
* `set_max_background_jobs(6)` and `set_max_subcompactions(2)`
* Block-based table: bloom filter (10), `cache_index_and_filter_blocks(true)`
* If you scan by height often, add a **prefix extractor** for height keyspace.

# Tiny test matrix (catch regressions)

* ✅ Graceful stop while blocks are flowing → shutdown ≤ 15s, miners reconnect.
* ✅ Persist timeout injection → 3 retries then **producer paused**, API stays up.
* ✅ Mining queue saturation → 429 surfaced; memory flat.
* ✅ Height cache TTL correctness → cache-hit reads < 1µs, full recompute ≥ TTL.

# What “good” looks like for the next 4+ hours

* Height advances every 2–5s.
* `p99(save_block)` < 500 ms; no executor starvation.
* No “binary search” logs during shutdown.
* Uptime crosses **4h** with zero manual restarts.

# Final note

You’ve proven the diagnosis end-to-end: external miners fixed the systemic stalls, and you captured hard evidence of the shutdown storm. Ship the HeightState wiring + shutdown broadcast now, then the timeouts/actor this afternoon. That combo will turn today’s success into **boring**, durable reliability. 🚀
