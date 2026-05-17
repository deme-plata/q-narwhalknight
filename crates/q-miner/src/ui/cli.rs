//! Command-line interface for the miner

use anyhow::Result;
use console::style;
use tracing::info;

pub struct CLIInterface {
    interactive: bool,
}

impl CLIInterface {
    pub fn new(interactive: bool) -> Self {
        Self { interactive }
    }
    
    pub async fn start(&self) -> Result<()> {
        if self.interactive {
            self.show_interactive_interface().await
        } else {
            self.show_simple_stats().await
        }
    }
    
    async fn show_interactive_interface(&self) -> Result<()> {
        info!("🖥️  Starting interactive CLI interface");
        
        // TODO: Implement interactive CLI
        println!("{}", style("📊 Quillon Miner").green().bold());
        println!("Interactive mode not yet implemented");
        Ok(())
    }
    
    async fn show_simple_stats(&self) -> Result<()> {
        info!("📊 Mining stats display");
        
        // TODO: Implement stats display
        println!("Mining statistics:");
        println!("  Hash rate: 0 H/s");
        println!("  Shares: 0");
        Ok(())
    }
}