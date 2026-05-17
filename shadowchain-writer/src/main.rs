use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;

mod cli;
mod database;
mod entities;
mod latex;
mod story;
mod ai;
mod analysis;

use cli::*;
use database::StoryDatabase;
use story::StoryCommands;

#[derive(Parser)]
#[command(name = "shadowchain")]
#[command(about = "A sophisticated CLI for crafting cyberpunk noir novels with entity tracking")]
#[command(version = "0.1.0")]
#[command(author = "Q-NarwhalKnight Authors")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Database path
    #[arg(short, long, default_value = "./story.db")]
    database: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new story project
    Init {
        /// Story title
        title: String,
        /// Author name
        #[arg(short, long)]
        author: Option<String>,
    },

    /// Entity management commands
    Entity {
        #[command(subcommand)]
        action: EntityCommands,
    },

    /// Chapter and scene management
    Story {
        #[command(subcommand)]
        action: StoryCommands,
    },

    /// Generate LaTeX and PDF output
    Export {
        #[command(subcommand)]
        format: ExportCommands,
    },

    /// Analyze story structure and entities
    Analyze {
        #[command(subcommand)]
        analysis: AnalysisCommands,
    },

    /// Interactive story dashboard
    Dashboard,

    /// AI-assisted writing tools
    #[cfg(feature = "ai")]
    AI {
        #[command(subcommand)]
        tool: AICommands,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize colored output
    if !console::Term::stdout().features().colors_supported() {
        colored::control::set_override(false);
    }

    // Initialize database
    let mut db = StoryDatabase::new(&cli.database).await?;

    match cli.command {
        Commands::Init { title, author } => {
            init_story(&mut db, title, author).await?;
        }
        Commands::Entity { action } => {
            handle_entity_command(&mut db, action).await?;
        }
        Commands::Story { action } => {
            handle_story_command(&mut db, action).await?;
        }
        Commands::Export { format } => {
            handle_export_command(&mut db, format).await?;
        }
        Commands::Analyze { analysis } => {
            handle_analysis_command(&mut db, analysis).await?;
        }
        Commands::Dashboard => {
            launch_dashboard(&mut db).await?;
        }
        #[cfg(feature = "ai")]
        Commands::AI { tool } => {
            handle_ai_command(&mut db, tool).await?;
        }
    }

    Ok(())
}

async fn init_story(db: &mut StoryDatabase, title: String, author: Option<String>) -> Result<()> {
    println!("{}", "🖋️  Initializing Shadows in the Chain story project...".bright_cyan());

    let story_id = db.create_story(&title, author.as_deref()).await?;

    // Pre-populate with main entities from your story
    let elena_id = db.create_entity(story_id, entities::EntityType::Character {
        name: "Elena Voss".to_string(),
        description: "MI6 operative burned twice over, ghost in the machine".to_string(),
        traits: vec!["scarred".to_string(), "weary".to_string(), "dangerous".to_string()],
        relationships: vec![],
    }).await?;

    let hale_id = db.create_entity(story_id, entities::EntityType::Character {
        name: "Marcus Hale".to_string(),
        description: "CIA analyst, crypto-hunter obsessed with taking down Elena".to_string(),
        traits: vec!["obsessive".to_string(), "analytical".to_string(), "haunted".to_string()],
        relationships: vec![entities::Relationship {
            target_id: elena_id,
            relationship_type: "nemesis".to_string(),
            description: "Personal vendetta after losing an asset to Elena".to_string(),
        }],
    }).await?;

    db.create_entity(story_id, entities::EntityType::Technology {
        name: "Nexus Veil".to_string(),
        description: "Underground protocol for anonymous blockchain transactions".to_string(),
        technical_details: vec![
            "SNARKs and STARKs for zero-knowledge proofs".to_string(),
            "IPFS for content-addressed storage".to_string(),
            "BitTorrent DHT for peer discovery".to_string(),
            "BEP-44 for soulbound tokens".to_string(),
        ],
    }).await?;

    db.create_entity(story_id, entities::EntityType::Location {
        name: "Berlin Kreuzberg".to_string(),
        description: "Graffiti-scarred district, rain-slicked streets".to_string(),
        atmosphere: "Noir, industrial decay".to_string(),
        significance: vec!["Elena's safe haven".to_string(), "Story opening".to_string()],
    }).await?;

    db.create_entity(story_id, entities::EntityType::MacGuffin {
        name: "Master Node".to_string(),
        description: "Genesis key that can rewrite Nexus Veil's rules".to_string(),
        importance: "Control over global anonymity infrastructure".to_string(),
        location: "Cold War bunker beneath Warsaw".to_string(),
    }).await?;

    println!("{}", "✅ Story initialized with core entities!".bright_green());
    println!("   📖 Title: {}", title.bright_white());
    if let Some(author) = author {
        println!("   ✍️  Author: {}", author.bright_white());
    }
    println!("   🗃️  Database: {}", db.path().bright_yellow());

    Ok(())
}