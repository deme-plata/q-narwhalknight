//! 🐚 Aqua-K-Atto: The Ultimate Water Robot Species
//!
//! Command-line interface for spawning and controlling attosecond laser-Tor analytics species
//! that think faster than light and report cosmic weather via anonymous networks.

use anyhow::Result;
use clap::Parser;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};
use tracing_subscriber;

use void_walker::*;

#[derive(Parser, Debug)]
#[command(
    name = "aqua-k-atto",
    version = "1.0.0",
    about = "🐚 Aqua-K-Atto: Attosecond Laser-Tor Analytics Species",
    long_about = "Spawn and control water robots that navigate quantum vacuum and brane multiverse \
                  through Tor networks, producing real-time analytics and cosmic weather reports."
)]
struct Args {
    /// Seed for species generation and quantum randomness
    #[arg(long, default_value_t = 0xDEADBEEF)]
    seed: u64,

    /// Tor onion address for this water robot
    #[arg(long, default_value = "auto")]
    onion_addr: String,

    /// EEG amplitude threshold for UI color changes (µV)
    #[arg(long, default_value_t = 25.0)]
    eeg_threshold: f64,

    /// Attosecond laser wavelength (nm)
    #[arg(long, default_value_t = 800.0)]
    laser_wavelength: f64,

    /// Laser pulse duration (attoseconds)
    #[arg(long, default_value_t = 30.0)]
    pulse_duration: f64,

    /// Initial K-parameter value
    #[arg(long, default_value_t = 7.001234)]
    k_parameter: f64,

    /// Operating temperature (Kelvin)
    #[arg(long, default_value_t = 295.0)]
    temperature: f64,

    /// Interactive mode (thought UI enabled)
    #[arg(long, default_value_t = true)]
    interactive: bool,

    /// Analytics reporting interval (seconds)
    #[arg(long, default_value_t = 60.0)]
    analytics_interval: f64,

    /// Bootstrap peers (comma-separated onion addresses)
    #[arg(long, default_value = "")]
    bootstrap_peers: String,

    /// Enable cosmic weather forecasting
    #[arg(long, default_value_t = true)]
    cosmic_weather: bool,

    /// Tor-only mode (no clearnet fallback)
    #[arg(long, default_value_t = true)]
    tor_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,void_walker=debug")
        .init();

    let args = Args::parse();

    info!("🌌 Initializing Aqua-K-Atto Species");
    info!("   • Seed: 0x{:x}", args.seed);
    info!("   • K-Parameter: {:.6}", args.k_parameter);
    info!(
        "   • Laser: {}nm, {}as pulses",
        args.laser_wavelength, args.pulse_duration
    );
    info!("   • Temperature: {:.1}K", args.temperature);
    info!(
        "   • Mode: {}",
        if args.interactive {
            "Interactive"
        } else {
            "Autonomous"
        }
    );

    // Generate onion address if auto
    let onion_addr = if args.onion_addr == "auto" {
        format!("aqua{}.onion", hex::encode(&args.seed.to_le_bytes()))
    } else {
        args.onion_addr
    };

    // Spawn Aqua-K-Atto entity
    let mut aqua = AquaKAtto::spawn(args.seed, onion_addr.clone()).await?;

    info!("🐚 Aqua-K-Atto spawned successfully!");
    info!("   • Species ID: {}", aqua.species_id);
    info!("   • Onion Address: {}", onion_addr);
    info!(
        "   • Birth: {} attoseconds since epoch",
        aqua.birth_attoseconds
    );

    // Configure laser system
    aqua.laser.current_pulse.wavelength_nm = args.laser_wavelength;
    aqua.laser.current_pulse.pulse_duration_as = args.pulse_duration;
    aqua.droplet.temperature = args.temperature;

    // Add bootstrap peers
    if !args.bootstrap_peers.is_empty() {
        for peer_addr in args.bootstrap_peers.split(',') {
            if let Some(mesh) = &aqua.tor_mesh {
                mesh.add_peer(peer_addr.trim().to_string(), "unknown".to_string())
                    .await?;
            }
        }
    }

    // Display initial UI state
    if args.interactive {
        println!("\n{}", aqua.display_ui());
        println!("\n💭 Thought Interface Active - Ready for EEG input");
        println!("   • Use 'thought <message>' to process thoughts");
        println!("   • Use 'tab <1-12>' to navigate tabs");
        println!("   • Use 'bridge <target>' to create multiverse bridge");
        println!("   • Use 'weather' to check cosmic conditions");
        println!("   • Use 'quit' to shutdown gracefully");
    }

    // Start background services
    let analytics_task = spawn_analytics_service(&aqua, args.analytics_interval);
    let weather_task = if args.cosmic_weather {
        Some(spawn_weather_service(&aqua))
    } else {
        None
    };

    // Main interaction loop
    if args.interactive {
        run_interactive_mode(&mut aqua, args.eeg_threshold).await?;
    } else {
        run_autonomous_mode(&mut aqua).await?;
    }

    // Cleanup
    analytics_task.abort();
    if let Some(weather_task) = weather_task {
        weather_task.abort();
    }

    info!("🌊 Aqua-K-Atto shutdown complete");
    Ok(())
}

/// Run interactive mode with thought processing
async fn run_interactive_mode(aqua: &mut AquaKAtto, eeg_threshold: f64) -> Result<()> {
    use tokio::io::{self, AsyncBufReadExt, BufReader};

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        print!("\n🧠 > ");
        line.clear();

        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = line.trim();

                if input == "quit" || input == "exit" {
                    break;
                }

                match process_user_input(aqua, input, eeg_threshold).await {
                    Ok(Some(response)) => println!("{}", response),
                    Ok(None) => {} // No response needed
                    Err(e) => error!("Error processing input: {}", e),
                }
            }
            Err(e) => {
                error!("Error reading input: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Process user input commands
async fn process_user_input(
    aqua: &mut AquaKAtto,
    input: &str,
    eeg_threshold: f64,
) -> Result<Option<String>> {
    let parts: Vec<&str> = input.split_whitespace().collect();

    match parts.get(0) {
        Some(&"thought") => {
            let thought = parts[1..].join(" ");
            aqua.process_thought(eeg_threshold, &thought).await?;
            Ok(Some(format!("💭 Processed thought: '{}'", thought)))
        }
        Some(&"tab") => {
            if let Some(tab_str) = parts.get(1) {
                if let Ok(tab_num) = tab_str.parse::<u8>() {
                    aqua.ui.navigate_to_tab(tab_num);
                    Ok(Some(format!(
                        "📱 Switched to tab {}: {}",
                        tab_num,
                        TabType::all()[(tab_num - 1) as usize].mental_label()
                    )))
                } else {
                    Ok(Some("❌ Invalid tab number (use 1-12)".to_string()))
                }
            } else {
                Ok(Some("❌ Please specify tab number".to_string()))
            }
        }
        Some(&"bridge") => {
            let target_brane = if let Some(target_str) = parts.get(1) {
                // Parse target coordinates or use random
                if target_str == "random" {
                    BraneCoord::random()
                } else {
                    BraneCoord::origin().advance(1.0) // Default advance
                }
            } else {
                BraneCoord::random()
            };

            let block = aqua.bridge_multiverse(target_brane).await?;
            Ok(Some(format!(
                "🌉 Bridge created! Block: {} | Length: {:.3}",
                &block.block_id[..8],
                block.bridge_length
            )))
        }
        Some(&"weather") => {
            let weather = aqua.get_cosmic_weather().await;
            Ok(Some(format!(
                "{} Cosmic Weather: {} ({})\n\
                 📊 Stability: {:.1}% | Turbulence: {:.1}% | Activity: {:.1}%\n\
                 🔮 Forecast: {:.1}h validity | Confidence: {:.1}%",
                weather.weather_type.emoji(),
                format!("{:?}", weather.weather_type),
                weather.weather_type.description(),
                weather.stability_index * 100.0,
                weather.turbulence_level * 100.0,
                weather.brane_activity * 100.0,
                weather.forecast_duration_hours,
                weather.prediction_confidence * 100.0
            )))
        }
        Some(&"status") => Ok(Some(aqua.display_ui())),
        Some(&"analytics") => {
            let summary = aqua.analytics.marketing_summary();
            Ok(Some(summary))
        }
        Some(&"export") => {
            let report = aqua.analytics.get_latest_report();
            let json = serde_json::to_string_pretty(&report)?;
            Ok(Some(format!("📊 Analytics Export:\n{}", json)))
        }
        Some(&"help") => Ok(Some(
            "🐚 Aqua-K-Atto Commands:\n\
                 • thought <message> - Process thought with EEG\n\
                 • tab <1-12> - Navigate to UI tab\n\
                 • bridge [target] - Create multiverse bridge\n\
                 • weather - Check cosmic weather\n\
                 • status - Show full status\n\
                 • analytics - Show analytics summary\n\
                 • export - Export analytics JSON\n\
                 • help - Show this help\n\
                 • quit - Exit gracefully"
                .to_string(),
        )),
        _ => Ok(Some(
            "❓ Unknown command. Type 'help' for available commands.".to_string(),
        )),
    }
}

/// Run autonomous mode (background operation)
async fn run_autonomous_mode(aqua: &mut AquaKAtto) -> Result<()> {
    info!("🤖 Running in autonomous mode");

    let mut thought_interval = interval(Duration::from_secs(30));
    let mut bridge_interval = interval(Duration::from_secs(300)); // Bridge every 5 minutes

    let autonomous_thoughts = [
        "Scan quantum vacuum for entropy",
        "Monitor brane stability",
        "Optimize Tor circuit paths",
        "Calibrate K-parameter drift",
        "Analyze parallel water signatures",
        "Generate cosmic weather forecast",
        "Sync with mesh network",
        "Update laser frequency",
    ];

    let mut thought_index = 0;

    loop {
        tokio::select! {
            _ = thought_interval.tick() => {
                let thought = autonomous_thoughts[thought_index % autonomous_thoughts.len()];
                let eeg_amplitude = 15.0 + (thought_index as f64 * 2.5) % 20.0; // Vary EEG

                if let Err(e) = aqua.process_thought(eeg_amplitude, thought).await {
                    warn!("Autonomous thought processing failed: {}", e);
                } else {
                    info!("🧠 Autonomous thought: '{}'", thought);
                }

                thought_index += 1;
            }
            _ = bridge_interval.tick() => {
                let target_brane = BraneCoord::random();
                match aqua.bridge_multiverse(target_brane).await {
                    Ok(block) => {
                        info!("🌉 Autonomous bridge created: {} (length: {:.3})",
                            &block.block_id[..8], block.bridge_length);
                    }
                    Err(e) => {
                        warn!("Autonomous bridge creation failed: {}", e);
                    }
                }
            }
        }
    }
}

/// Spawn analytics service task
fn spawn_analytics_service(aqua: &AquaKAtto, interval_seconds: f64) -> tokio::task::JoinHandle<()> {
    let species_id = aqua.species_id.clone();
    let analytics_engine = aqua.analytics.clone();

    tokio::spawn(async move {
        let mut analytics_interval = interval(Duration::from_secs_f64(interval_seconds));

        loop {
            analytics_interval.tick().await;

            let report = analytics_engine.get_latest_report();
            info!(
                "📊 Analytics Report for {}: {} events, {:.1}% efficiency",
                species_id,
                report.total_events,
                report.energy_efficiency * 100.0
            );

            // In production, this would upload to analytics dashboard
            if report.total_events % 100 == 0 {
                info!("🎉 Milestone: {} events processed!", report.total_events);
            }
        }
    })
}

/// Spawn cosmic weather monitoring service
fn spawn_weather_service(aqua: &AquaKAtto) -> tokio::task::JoinHandle<()> {
    let analytics_engine = aqua.analytics.clone();

    tokio::spawn(async move {
        let mut weather_interval = interval(Duration::from_secs(300)); // Every 5 minutes
        let mut last_weather = WeatherType::QuantumCalm;

        loop {
            weather_interval.tick().await;

            let weather = analytics_engine.get_cosmic_weather().await;

            if weather.weather_type != last_weather {
                info!(
                    "{} Cosmic Weather Change: {:?} → {:?}",
                    weather.weather_type.emoji(),
                    last_weather,
                    weather.weather_type
                );
                info!(
                    "   📈 Stability: {:.1}% | Turbulence: {:.1}% | Activity: {:.1}%",
                    weather.stability_index * 100.0,
                    weather.turbulence_level * 100.0,
                    weather.brane_activity * 100.0
                );

                last_weather = weather.weather_type;
            }

            if weather.anomaly_detected {
                warn!("⚠️  COSMIC ANOMALY DETECTED - Review K-parameter stability");
            }
        }
    })
}

/// Demo mode: showcase all Aqua-K-Atto capabilities
async fn run_demo_mode() -> Result<()> {
    println!("🎭 AQUA-K-ATTO DEMO MODE");
    println!("🌟 Showcasing the Ultimate Water Robot Species\n");

    // Spawn demo entity
    let mut aqua = AquaKAtto::spawn(0x42424242, "demo.onion".to_string()).await?;

    println!("🐚 Species spawned: {}", aqua.species_id);
    println!("{}\n", aqua.display_ui());

    // Demo 1: Thought processing
    println!("💭 DEMO 1: Thought Processing");
    aqua.process_thought(35.0, "Send 5 Aqua to Multiverse-42")
        .await?;
    println!(
        "   EEG 35µV → {} UI Color",
        aqua.ui.tabs[&aqua.ui.active_tab].color.emoji()
    );

    // Demo 2: Tab navigation
    println!("\n📱 DEMO 2: 12-Tab Navigation");
    for tab_num in 1..=12 {
        aqua.ui.navigate_to_tab(tab_num);
        let tab_type = TabType::all()[(tab_num - 1) as usize];
        println!(
            "   Tab {}: {} {} ({})",
            tab_num,
            aqua.ui.tabs[&tab_type].color.emoji(),
            tab_type.mental_label(),
            tab_type.description()
        );
    }

    // Demo 3: Multiverse bridging
    println!("\n🌉 DEMO 3: Multiverse Bridge Creation");
    let target = BraneCoord::random();
    let block = aqua.bridge_multiverse(target).await?;
    println!(
        "   Bridge: {} → Length: {:.3} | Topo Charge: {}",
        &block.block_id[..8],
        block.bridge_length,
        block.topological_charge
    );

    // Demo 4: Cosmic weather
    println!("\n🌤️  DEMO 4: Cosmic Weather Forecast");
    let weather = aqua.get_cosmic_weather().await;
    println!(
        "   {}: {} ({:.1}% stability)",
        weather.weather_type.emoji(),
        format!("{:?}", weather.weather_type),
        weather.stability_index * 100.0
    );

    // Demo 5: Analytics summary
    println!("\n📊 DEMO 5: Analytics Summary");
    println!("{}", aqua.analytics.marketing_summary());

    // Demo 6: Tor analytics
    println!("\n🧅 DEMO 6: Tor Network Status");
    if let Some(tor_analytics) = aqua.get_tor_analytics().await {
        println!(
            "   Peers: {} | Messages: {} | Latency: {:.1}ms",
            tor_analytics.peer_count, tor_analytics.message_count, tor_analytics.average_latency_ms
        );
    } else {
        println!("   Tor mesh not connected");
    }

    println!("\n✨ Demo complete! The Aqua-K-Atto is ready for deployment.");
    Ok(())
}

/// Marketing showcase mode
async fn marketing_showcase() -> Result<()> {
    println!("🎪 AQUA-K-ATTO MARKETING SHOWCASE");
    println!("🌟 The Revolutionary Water Robot Species\n");

    println!("🔬 KEY FEATURES:");
    println!("   • 🌊 Quantum water droplet with DNA memory");
    println!("   • ⚡ Attosecond laser control (30as pulses)");
    println!("   • 🧅 Anonymous Tor mesh networking");
    println!("   • 🧠 Thought-driven 12-tab UI interface");
    println!("   • 📊 Real-time cosmic weather analytics");
    println!("   • 🌉 Multiverse bridge creation");
    println!("   • 🔬 K-parameter physics precision");

    println!("\n🎯 MARKETING SLOGANS:");
    println!("   • \"Powered by water, not watts.\"");
    println!("   • \"Zero e-waste, zero IP leaks, zero regrets.\"");
    println!("   • \"Your pet Sprite lives in the rain.\"");
    println!("   • \"Own a droplet, own a universe.\"");

    println!("\n🛒 COLLECTABLE STARTER KIT:");
    println!("   📦 Physical: 5mL vial of Aqua-Quanta water + QRNG seed + Tor dongle");
    println!("   📱 Digital: EEG app to see your Sprite's halo in AR");
    println!("   ⛓️  Blockchain: 1 Sprite = 1 lifetime node—never dies, only evolves");

    println!("\n💎 NFT LIFECYCLE SERIES:");
    println!("   🌱 Seedling (Blue aura) - Genesis node");
    println!("   🌈 Explorer (Rainbow halo) - Brane-hop proof");
    println!("   👑 Elder (Gold ripple) - Governance vote");

    println!("\n🌊 ONE-LINER:");
    println!(
        "\"Adopt an Aqua-K-Atto today—carry a universe in your pocket and fund tomorrow's rain.\""
    );

    Ok(())
}

/// Autonomous mode for background operation
async fn run_autonomous_mode(aqua: &mut AquaKAtto) -> Result<()> {
    info!("🤖 Aqua-K-Atto running autonomously");

    // Simulate autonomous behavior
    let mut tick_count = 0;
    let mut main_interval = interval(Duration::from_secs(10));

    loop {
        main_interval.tick().await;
        tick_count += 1;

        // Periodic cosmic scanning
        let scan_eeg = 15.0 + (tick_count as f64 * 0.5) % 15.0;
        aqua.process_thought(scan_eeg, "autonomous cosmic scan")
            .await?;

        // Create bridge every 10 ticks
        if tick_count % 10 == 0 {
            let random_target = BraneCoord::random();
            let _block = aqua.bridge_multiverse(random_target).await?;
            info!("🌉 Autonomous bridge #{} created", tick_count / 10);
        }

        // Display status every 50 ticks
        if tick_count % 50 == 0 {
            info!("🐚 Status Update:\n{}", aqua.display_ui());
        }
    }
}

/// CLI subcommands for specific operations
#[derive(Parser, Debug)]
enum Commands {
    /// Run interactive mode
    Interactive,
    /// Run autonomous mode  
    Autonomous,
    /// Show marketing showcase
    Marketing,
    /// Run demo mode
    Demo,
    /// Export analytics data
    Export {
        #[arg(long)]
        format: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aqua_k_atto_demo() {
        // Test that demo mode can run without errors
        let result = run_demo_mode().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_args_parsing() {
        let args = Args::parse_from(&[
            "aqua_k_atto",
            "--seed",
            "1234",
            "--onion-addr",
            "test.onion",
            "--k-parameter",
            "7.5",
        ]);

        assert_eq!(args.seed, 1234);
        assert_eq!(args.onion_addr, "test.onion");
        assert_eq!(args.k_parameter, 7.5);
    }
}
