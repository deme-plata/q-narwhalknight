/// Executable runner for Q-NarwhalKnight Performance Demo

use anyhow::Result;

mod simple_performance_demo;
use simple_performance_demo::run_5_node_performance_demo;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize basic logging
    env_logger::init();
    
    // Run the performance demonstration
    run_5_node_performance_demo().await?;
    
    Ok(())
}