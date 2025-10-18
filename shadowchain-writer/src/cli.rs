use anyhow::Result;
use clap::Subcommand;
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select, Input, Confirm, MultiSelect};
use uuid::Uuid;

use crate::database::StoryDatabase;
use crate::entities::{EntityType, EntityFilter, templates};
use crate::story::StoryCommands;
use crate::latex::LaTeXGenerator;
use crate::analysis::StoryAnalyzer;

#[derive(Subcommand)]
pub enum EntityCommands {
    /// Create a new entity (interactive)
    Create {
        /// Entity type (character, tech, location, macguffin, org)
        #[arg(short, long)]
        entity_type: String,
        /// Entity name
        name: String,
    },
    /// Create entity from JSON file (non-interactive)
    CreateFrom {
        /// Path to JSON file with entity data
        json_file: String,
    },
    /// Batch create entities from JSON array file
    BatchCreate {
        /// Path to JSON file with array of entities
        json_file: String,
    },
    /// List entities with optional filtering
    List {
        /// Filter by entity type
        #[arg(short, long)]
        entity_type: Option<String>,
        /// Filter by tag
        #[arg(short, long)]
        tag: Option<String>,
    },
    /// Show detailed entity information
    Show {
        /// Entity name or ID
        entity: String,
    },
    /// Edit an existing entity
    Edit {
        /// Entity name or ID
        entity: String,
    },
    /// Delete an entity
    Delete {
        /// Entity name or ID
        entity: String,
    },
    /// Search entities
    Search {
        /// Search query
        query: String,
    },
    /// Show entity relationships graph
    Graph,
    /// Interactive entity browser
    Browse,
}

#[derive(Subcommand)]
pub enum ExportCommands {
    /// Generate LaTeX source
    LaTeX {
        /// Output file path
        #[arg(short, long, default_value = "shadowchain.tex")]
        output: String,
    },
    /// Generate PDF
    PDF {
        /// Output file path
        #[arg(short, long, default_value = "shadowchain.pdf")]
        output: String,
    },
    /// Export entity database as JSON
    Entities {
        /// Output file path
        #[arg(short, long, default_value = "entities.json")]
        output: String,
    },
}

#[derive(Subcommand)]
pub enum AnalysisCommands {
    /// Show story statistics
    Stats,
    /// Analyze entity relationships
    Relationships,
    /// Show story structure
    Structure,
    /// Generate entity analytics
    EntityReport,
    /// Timeline visualization
    Timeline,
}

#[cfg(feature = "ai")]
#[derive(Subcommand)]
pub enum AICommands {
    /// Generate character description
    Character {
        /// Character name
        name: String,
    },
    /// Suggest plot developments
    Plot,
    /// Analyze writing style
    Style,
    /// Generate tech descriptions
    Tech {
        /// Technology name
        name: String,
    },
}

pub async fn handle_entity_command(db: &mut StoryDatabase, command: EntityCommands) -> Result<()> {
    match command {
        EntityCommands::Create { entity_type, name } => {
            create_entity_interactive(db, &entity_type, &name).await?;
        }
        EntityCommands::CreateFrom { json_file } => {
            create_entity_from_json(db, &json_file).await?;
        }
        EntityCommands::BatchCreate { json_file } => {
            batch_create_entities(db, &json_file).await?;
        }
        EntityCommands::List { entity_type, tag } => {
            list_entities(db, entity_type, tag).await?;
        }
        EntityCommands::Show { entity } => {
            show_entity(db, &entity).await?;
        }
        EntityCommands::Edit { entity } => {
            edit_entity_interactive(db, &entity).await?;
        }
        EntityCommands::Delete { entity } => {
            delete_entity_interactive(db, &entity).await?;
        }
        EntityCommands::Search { query } => {
            search_entities(db, &query).await?;
        }
        EntityCommands::Graph => {
            show_relationship_graph(db).await?;
        }
        EntityCommands::Browse => {
            browse_entities_interactive(db).await?;
        }
    }
    Ok(())
}

pub async fn handle_story_command(db: &mut StoryDatabase, command: StoryCommands) -> Result<()> {
    // Implementation for story commands
    Ok(())
}

pub async fn handle_export_command(db: &mut StoryDatabase, command: ExportCommands) -> Result<()> {
    match command {
        ExportCommands::LaTeX { output } => {
            generate_latex(db, &output).await?;
        }
        ExportCommands::PDF { output } => {
            generate_pdf(db, &output).await?;
        }
        ExportCommands::Entities { output } => {
            export_entities_json(db, &output).await?;
        }
    }
    Ok(())
}

pub async fn handle_analysis_command(db: &mut StoryDatabase, command: AnalysisCommands) -> Result<()> {
    let analyzer = StoryAnalyzer::new();

    match command {
        AnalysisCommands::Stats => {
            analyzer.show_story_stats(db).await?;
        }
        AnalysisCommands::Relationships => {
            analyzer.show_relationship_analysis(db).await?;
        }
        AnalysisCommands::Structure => {
            analyzer.show_story_structure(db).await?;
        }
        AnalysisCommands::EntityReport => {
            analyzer.generate_entity_report(db).await?;
        }
        AnalysisCommands::Timeline => {
            analyzer.show_timeline(db).await?;
        }
    }
    Ok(())
}

#[cfg(feature = "ai")]
pub async fn handle_ai_command(db: &mut StoryDatabase, command: AICommands) -> Result<()> {
    // AI-assisted writing tools implementation
    Ok(())
}

pub async fn launch_dashboard(db: &mut StoryDatabase) -> Result<()> {
    println!("{}", "🎬 Shadows in the Chain - Story Dashboard".bright_cyan().bold());

    loop {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            println!("{}", "No stories found. Create one first with 'shadowchain init'".yellow());
            return Ok(());
        }

        let story = &stories[0]; // For now, use first story
        let entities = db.list_entities(story.id, None).await?;
        let chapters = db.list_chapters(story.id).await?;

        println!("\n{}", format!("📖 {} by {}",
            story.title.bright_white().bold(),
            story.author.as_deref().unwrap_or("Unknown").bright_white()
        ));

        println!("{}", format!("   📊 {} entities, {} chapters",
            entities.len().to_string().bright_green(),
            chapters.len().to_string().bright_green()
        ));

        let actions = vec![
            "Browse Entities",
            "View Chapters",
            "Entity Analytics",
            "Export to PDF",
            "Relationship Graph",
            "Exit",
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("What would you like to do?")
            .items(&actions)
            .interact()?;

        match selection {
            0 => browse_entities_interactive(db).await?,
            1 => browse_chapters_interactive(db, story.id).await?,
            2 => show_entity_analytics(db, story.id).await?,
            3 => generate_pdf(db, "shadowchain.pdf").await?,
            4 => show_relationship_graph(db).await?,
            5 => break,
            _ => {}
        }
    }

    Ok(())
}

async fn create_entity_interactive(db: &mut StoryDatabase, entity_type: &str, name: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        println!("{}", "No stories found. Create one first.".red());
        return Ok(());
    }

    let story = &stories[0]; // Use first story for now

    let entity_type = match entity_type.to_lowercase().as_str() {
        "character" | "char" => {
            let description: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Character description")
                .interact_text()?;

            let traits_input: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Character traits (comma-separated)")
                .interact_text()?;
            let traits: Vec<String> = traits_input
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            EntityType::Character {
                name: name.to_string(),
                description,
                traits,
                relationships: vec![],
            }
        }
        "tech" | "technology" => {
            let description: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Technology description")
                .interact_text()?;

            let technical_input: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Technical details (comma-separated)")
                .interact_text()?;
            let technical_details: Vec<String> = technical_input
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            EntityType::Technology {
                name: name.to_string(),
                description,
                technical_details,
            }
        }
        "location" | "loc" => {
            let description: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Location description")
                .interact_text()?;

            let atmosphere: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Atmosphere")
                .interact_text()?;

            EntityType::Location {
                name: name.to_string(),
                description,
                atmosphere,
                significance: vec![],
            }
        }
        "macguffin" => {
            let description: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("MacGuffin description")
                .interact_text()?;

            let importance: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Why is it important?")
                .interact_text()?;

            let location: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Current location")
                .interact_text()?;

            EntityType::MacGuffin {
                name: name.to_string(),
                description,
                importance,
                location,
            }
        }
        "org" | "organization" => {
            let description: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Organization description")
                .interact_text()?;

            let agenda: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Organization agenda")
                .interact_text()?;

            EntityType::Organization {
                name: name.to_string(),
                description,
                members: vec![],
                agenda,
            }
        }
        _ => {
            println!("{}", "Unknown entity type. Use: character, tech, location, macguffin, org".red());
            return Ok(());
        }
    };

    let entity_id = db.create_entity(story.id, entity_type).await?;
    println!("{}", format!("✅ Created entity: {} ({})", name.bright_white(), entity_id).green());

    Ok(())
}

async fn list_entities(db: &mut StoryDatabase, entity_type: Option<String>, tag: Option<String>) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        println!("{}", "No stories found.".red());
        return Ok(());
    }

    let story = &stories[0];
    let mut filter = EntityFilter::new();

    if let Some(et) = entity_type {
        filter = filter.with_type(&et);
    }

    if let Some(t) = tag {
        filter = filter.with_tag(&t);
    }

    let entities = db.list_entities(story.id, Some(filter)).await?;

    if entities.is_empty() {
        println!("{}", "No entities found matching the criteria.".yellow());
        return Ok(());
    }

    println!("{}", format!("📊 Found {} entities:", entities.len()).bright_cyan());

    for entity in entities {
        let type_color = match entity.entity_type_name() {
            "Character" => "bright_green",
            "Technology" => "bright_blue",
            "Location" => "bright_yellow",
            "MacGuffin" => "bright_magenta",
            "Organization" => "bright_red",
            _ => "white",
        };

        println!("  {} {} - {}",
            entity.entity_type_name().color(type_color).bold(),
            entity.name().bright_white().bold(),
            entity.description().dimmed()
        );
    }

    Ok(())
}

async fn show_entity(db: &mut StoryDatabase, entity_name: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        return Ok(());
    }

    let story = &stories[0];
    let entities = db.search_entities(story.id, entity_name).await?;

    if entities.is_empty() {
        println!("{}", format!("Entity '{}' not found.", entity_name).red());
        return Ok(());
    }

    let entity = &entities[0];

    println!("{}", format!("🎭 {} ({})", entity.name(), entity.entity_type_name()).bright_cyan().bold());
    println!("{}", format!("   {}", entity.description()).dimmed());

    // Show type-specific details
    match &entity.entity_type {
        crate::entities::EntityType::Character { traits, relationships, .. } => {
            if !traits.is_empty() {
                println!("   {} {}", "Traits:".bright_yellow(), traits.join(", "));
            }
            if !relationships.is_empty() {
                println!("   {} {}", "Relationships:".bright_yellow(), relationships.len());
            }
        }
        crate::entities::EntityType::Technology { technical_details, .. } => {
            if !technical_details.is_empty() {
                println!("   {} {}", "Technical Details:".bright_blue(), technical_details.join(", "));
            }
        }
        crate::entities::EntityType::Location { atmosphere, significance, .. } => {
            println!("   {} {}", "Atmosphere:".bright_green(), atmosphere);
            if !significance.is_empty() {
                println!("   {} {}", "Significance:".bright_green(), significance.join(", "));
            }
        }
        _ => {}
    }

    if !entity.tags.is_empty() {
        println!("   {} {}", "Tags:".bright_cyan(), entity.tags.join(", "));
    }

    Ok(())
}

// Additional helper functions would go here...

async fn browse_entities_interactive(db: &mut StoryDatabase) -> Result<()> {
    // Interactive entity browser implementation
    Ok(())
}

async fn browse_chapters_interactive(db: &mut StoryDatabase, story_id: Uuid) -> Result<()> {
    // Interactive chapter browser implementation
    Ok(())
}

async fn show_entity_analytics(db: &mut StoryDatabase, story_id: Uuid) -> Result<()> {
    let analytics = db.generate_entity_analytics(story_id).await?;

    println!("{}", "📊 Entity Analytics".bright_cyan().bold());
    println!("   {} {}", "Total Entities:".bright_white(), analytics.total_entities.to_string().bright_green());

    for (entity_type, count) in analytics.entity_counts {
        println!("   {} {}",
            format!("{}:", entity_type).bright_white(),
            count.to_string().bright_green()
        );
    }

    if let Some(character) = analytics.most_connected_character {
        println!("   {} {}", "Most Connected:".bright_white(), character.bright_yellow());
    }

    if !analytics.orphaned_entities.is_empty() {
        println!("   {} {}", "Orphaned Entities:".bright_red(), analytics.orphaned_entities.len());
    }

    Ok(())
}

async fn show_relationship_graph(db: &mut StoryDatabase) -> Result<()> {
    println!("{}", "🕸️  Relationship Graph (visualization coming soon...)".bright_cyan());
    Ok(())
}

async fn edit_entity_interactive(db: &mut StoryDatabase, entity_name: &str) -> Result<()> {
    // Entity editing implementation
    Ok(())
}

async fn delete_entity_interactive(db: &mut StoryDatabase, entity_name: &str) -> Result<()> {
    // Entity deletion implementation
    Ok(())
}

async fn search_entities(db: &mut StoryDatabase, query: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        return Ok(());
    }

    let story = &stories[0];
    let entities = db.search_entities(story.id, query).await?;

    if entities.is_empty() {
        println!("{}", format!("No entities found for '{}'.", query).yellow());
        return Ok(());
    }

    println!("{}", format!("🔍 Search results for '{}':", query).bright_cyan());

    for entity in entities {
        println!("  {} {} - {}",
            entity.entity_type_name().bright_blue().bold(),
            entity.name().bright_white().bold(),
            entity.description().dimmed()
        );
    }

    Ok(())
}

async fn generate_latex(db: &mut StoryDatabase, output: &str) -> Result<()> {
    let generator = LaTeXGenerator::new();
    generator.generate(db, output).await?;
    println!("{}", format!("✅ LaTeX generated: {}", output).green());
    Ok(())
}

async fn generate_pdf(db: &mut StoryDatabase, output: &str) -> Result<()> {
    let generator = LaTeXGenerator::new();
    generator.generate_pdf(db, output).await?;
    println!("{}", format!("✅ PDF generated: {}", output).green());
    Ok(())
}

async fn export_entities_json(db: &mut StoryDatabase, output: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        return Ok(());
    }

    let story = &stories[0];
    let entities = db.list_entities(story.id, None).await?;

    let json = serde_json::to_string_pretty(&entities)?;
    std::fs::write(output, json)?;

    println!("{}", format!("✅ Entities exported to: {}", output).green());
    Ok(())
}

// Non-interactive entity creation functions

async fn create_entity_from_json(db: &mut StoryDatabase, json_file: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        println!("{}", "No stories found. Create one first.".red());
        return Ok(());
    }

    let story = &stories[0];

    let json_data = std::fs::read_to_string(json_file)?;
    let entity_type: EntityType = serde_json::from_str(&json_data)?;

    let entity_id = db.create_entity(story.id, entity_type.clone()).await?;
    println!("{}", format!("✅ Created entity from JSON: {} ({})", entity_type.name(), entity_id).green());

    Ok(())
}

async fn batch_create_entities(db: &mut StoryDatabase, json_file: &str) -> Result<()> {
    let stories = db.list_stories().await?;
    if stories.is_empty() {
        println!("{}", "No stories found. Create one first.".red());
        return Ok(());
    }

    let story = &stories[0];

    let json_data = std::fs::read_to_string(json_file)?;
    let entities: Vec<EntityType> = serde_json::from_str(&json_data)?;

    println!("{}", format!("📦 Creating {} entities...", entities.len()).bright_cyan());

    for entity in entities {
        let entity_id = db.create_entity(story.id, entity.clone()).await?;
        println!("   ✓ {} ({})", entity.name().bright_white(), entity_id);
    }

    println!("{}", "✅ Batch creation complete!".bright_green());
    Ok(())
}