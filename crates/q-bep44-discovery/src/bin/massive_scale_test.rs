/*!
# BEP-44 Massive Scale Test Runner

This binary runs comprehensive massive scale tests for the BEP-44 DHT discovery system.
It can be used to evaluate Q-NarwhalKnight's peer discovery at various scales from small
networks to massive BitTorrent DHT integration.

## Usage Examples

```bash
# Quick test with 100 validators
cargo run --bin massive_scale_test -- --quick

# Medium scale test with 1K validators
cargo run --bin massive_scale_test -- --medium

# Full scale test with 10K validators and real DHT
cargo run --bin massive_scale_test -- --full --real-dht --real-tor

# Custom configuration
cargo run --bin massive_scale_test -- \
  --validators 5000 \
  --duration 600 \
  --churn-rate 0.15 \
  --byzantine 15 \
  --partitions
```

## Test Scenarios

1. **Quick Test**: 100 validators, 60 seconds, simulated components
2. **Medium Test**: 1K validators, 300 seconds, realistic churn
3. **Full Test**: 10K+ validators, real Tor + DHT integration
4. **Stress Test**: High churn, network partitions, Byzantine faults
*/

use clap::{Arg, Command};
use q_bep44_discovery::massive_scale_test::{
    run_full_massive_scale_test, run_quick_massive_scale_test, MassiveScaleTestSuite, TestConfig,
};
use tracing::{error, info};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    info!("🧪 BEP-44 Massive Scale Test Runner");
    info!("🌐 Testing Q-NarwhalKnight BitTorrent DHT + Tor Discovery");

    let matches = Command::new("massive_scale_test")
        .version("1.0")
        .about("BEP-44 massive scale discovery testing")
        .arg(
            Arg::new("quick")
                .long("quick")
                .action(clap::ArgAction::SetTrue)
                .help("Run quick test (100 validators, 60s)"),
        )
        .arg(
            Arg::new("medium")
                .long("medium")
                .action(clap::ArgAction::SetTrue)
                .help("Run medium test (1K validators, 300s)"),
        )
        .arg(
            Arg::new("full")
                .long("full")
                .action(clap::ArgAction::SetTrue)
                .help("Run full test (10K validators, 1800s)"),
        )
        .arg(
            Arg::new("validators")
                .long("validators")
                .value_name("COUNT")
                .help("Number of validators to simulate"),
        )
        .arg(
            Arg::new("duration")
                .long("duration")
                .value_name("SECONDS")
                .help("Test duration in seconds"),
        )
        .arg(
            Arg::new("churn-rate")
                .long("churn-rate")
                .value_name("RATE")
                .help("Validator churn rate (0-1)"),
        )
        .arg(
            Arg::new("byzantine")
                .long("byzantine")
                .value_name("PERCENT")
                .help("Byzantine validator percentage (0-33)"),
        )
        .arg(
            Arg::new("real-tor")
                .long("real-tor")
                .action(clap::ArgAction::SetTrue)
                .help("Use real Tor connections"),
        )
        .arg(
            Arg::new("real-dht")
                .long("real-dht")
                .action(clap::ArgAction::SetTrue)
                .help("Use real BitTorrent DHT"),
        )
        .arg(
            Arg::new("partitions")
                .long("partitions")
                .action(clap::ArgAction::SetTrue)
                .help("Simulate network partitions"),
        )
        .arg(
            Arg::new("geography")
                .long("geography")
                .action(clap::ArgAction::SetTrue)
                .help("Simulate geographic distribution"),
        )
        .get_matches();

    // Run test based on arguments
    let result = if matches.get_flag("quick") {
        info!("🚀 Running QUICK massive scale test");
        run_quick_massive_scale_test().await
    } else if matches.get_flag("medium") {
        info!("🚀 Running MEDIUM massive scale test");
        run_medium_massive_scale_test().await
    } else if matches.get_flag("full") {
        info!("🚀 Running FULL massive scale test");
        run_full_massive_scale_test().await
    } else {
        // Custom configuration
        let config = build_custom_config(&matches)?;
        info!("🚀 Running CUSTOM massive scale test");
        run_custom_massive_scale_test(config).await
    }?;

    // Print summary results
    info!("\n🎉 ================== TEST SUMMARY ==================");
    info!("✅ Test completed successfully!");
    info!(
        "📊 Total discoveries attempted: {}",
        result.total_discoveries_attempted
    );
    info!(
        "🎯 Successful discoveries: {}",
        result.successful_discoveries
    );
    info!(
        "📈 Success rate: {:.2}%",
        result.tor_connection_success_rate * 100.0
    );
    info!(
        "⏱️  Average discovery latency: {}ms",
        result.average_discovery_latency_ms
    );
    info!(
        "🛡️  Byzantine behavior detected: {}",
        result.byzantine_behavior_detected
    );

    if result.tor_connection_success_rate >= 0.95 {
        info!("🏆 EXCELLENT: Discovery success rate > 95%");
    } else if result.tor_connection_success_rate >= 0.90 {
        info!("✅ GOOD: Discovery success rate > 90%");
    } else if result.tor_connection_success_rate >= 0.80 {
        info!("⚠️  ACCEPTABLE: Discovery success rate > 80%");
    } else {
        error!("❌ POOR: Discovery success rate < 80%");
    }

    if result.average_discovery_latency_ms <= 15000 {
        info!("🚀 FAST: Average discovery latency < 15s");
    } else if result.average_discovery_latency_ms <= 30000 {
        info!("✅ GOOD: Average discovery latency < 30s");
    } else {
        info!("⚠️  SLOW: Average discovery latency > 30s");
    }

    info!("🌟 BEP-44 + Tor massive scale testing complete!");
    info!("====================================================\n");

    Ok(())
}

async fn run_medium_massive_scale_test(
) -> anyhow::Result<q_bep44_discovery::massive_scale_test::TestMetrics> {
    let config = TestConfig {
        validator_count: 1_000,
        test_duration_secs: 300, // 5 minutes
        churn_rate: 0.08,
        byzantine_percentage: 8.0,
        enable_real_tor: false,
        enable_real_dht: false,
        simulate_partitions: true,
        simulate_geography: true,
        ..TestConfig::default()
    };

    let mut test_suite = MassiveScaleTestSuite::new(config).await?;
    test_suite.initialize_validators().await?;
    test_suite.run_test_suite().await
}

async fn run_custom_massive_scale_test(
    config: TestConfig,
) -> anyhow::Result<q_bep44_discovery::massive_scale_test::TestMetrics> {
    let mut test_suite = MassiveScaleTestSuite::new(config).await?;
    test_suite.initialize_validators().await?;
    test_suite.run_test_suite().await
}

fn build_custom_config(matches: &clap::ArgMatches) -> anyhow::Result<TestConfig> {
    let mut config = TestConfig::default();

    if let Some(validators) = matches.get_one::<String>("validators") {
        config.validator_count = validators.parse()?;
    }

    if let Some(duration) = matches.get_one::<String>("duration") {
        config.test_duration_secs = duration.parse()?;
    }

    if let Some(churn_rate) = matches.get_one::<String>("churn-rate") {
        config.churn_rate = churn_rate.parse()?;
    }

    if let Some(byzantine) = matches.get_one::<String>("byzantine") {
        config.byzantine_percentage = byzantine.parse()?;
    }

    config.enable_real_tor = matches.get_flag("real-tor");
    config.enable_real_dht = matches.get_flag("real-dht");
    config.simulate_partitions = matches.get_flag("partitions");
    config.simulate_geography = matches.get_flag("geography");

    // Validate configuration
    if config.validator_count == 0 {
        anyhow::bail!("Validator count must be greater than 0");
    }

    if config.byzantine_percentage > 33.0 {
        anyhow::bail!("Byzantine percentage cannot exceed 33%");
    }

    if config.churn_rate < 0.0 || config.churn_rate > 1.0 {
        anyhow::bail!("Churn rate must be between 0 and 1");
    }

    info!("📋 Custom test configuration:");
    info!("   • Validators: {}", config.validator_count);
    info!("   • Duration: {}s", config.test_duration_secs);
    info!("   • Churn Rate: {:.2}", config.churn_rate);
    info!("   • Byzantine: {:.1}%", config.byzantine_percentage);
    info!("   • Real Tor: {}", config.enable_real_tor);
    info!("   • Real DHT: {}", config.enable_real_dht);

    Ok(config)
}
