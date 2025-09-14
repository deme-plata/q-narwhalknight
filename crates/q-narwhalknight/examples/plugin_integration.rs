use q_narwhalknight::plugin::DNSPhantomPlugin;
use q_plugin_system::{PluginSystem, PluginSystemConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("🔌 Starting DNS-Phantom plugin integration...");

    let config = PluginSystemConfig::default();
    let mut plugin_system = PluginSystem::new(config);

    plugin_system.add_plugin(Box::new(DNSPhantomPlugin::autonomous()));
    plugin_system.start().await?;

    println!("🎉 Plugin system started with DNS-Phantom mesh networking!");

    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    plugin_system.shutdown().await?;
    println!("🏁 Plugin integration demo completed!");

    Ok(())
}
