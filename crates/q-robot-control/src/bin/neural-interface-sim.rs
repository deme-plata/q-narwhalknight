/// Neural Interface Simulator Binary
///
/// Standalone binary for testing neural interface functionality
use anyhow::Result;
use q_robot_control::neural_interface::NeuralInterface;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Neural Interface Simulator Starting...");
    println!("🌊 Q-NarwhalKnight Water-Robot Neural Control System");

    // Initialize neural interface
    let _neural_interface = NeuralInterface::new().await?;

    println!("✅ Neural interface initialized successfully");
    println!("🔗 Ready for biological control signals");

    // Keep the simulation running
    tokio::signal::ctrl_c().await?;
    println!("🛑 Neural interface simulator shutting down...");

    Ok(())
}
