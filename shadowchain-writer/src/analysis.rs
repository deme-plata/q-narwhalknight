use anyhow::Result;
use colored::*;
use std::collections::HashMap;
use uuid::Uuid;

use crate::database::StoryDatabase;
use crate::entities::{Entity, EntityAnalytics};
use crate::story::{Story, Chapter, Scene, StoryStructure, Act, ThreeActAnalysis, PacingAnalysis, TensionPoint};

/// Story analysis and insights engine
pub struct StoryAnalyzer {
    analysis_cache: HashMap<Uuid, EntityAnalytics>,
}

impl StoryAnalyzer {
    pub fn new() -> Self {
        Self {
            analysis_cache: HashMap::new(),
        }
    }

    /// Display comprehensive story statistics
    pub async fn show_story_stats(&self, db: &StoryDatabase) -> Result<()> {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            println!("{}", "No stories found.".red());
            return Ok(());
        }

        let story = &stories[0];
        let entities = db.list_entities(story.id, None).await?;
        let chapters = db.list_chapters(story.id).await?;

        println!("{}", "📊 Story Statistics Dashboard".bright_cyan().bold());
        println!();

        // Story metadata
        println!("{}", "📖 Story Overview".bright_white().bold());
        println!("   {} {}", "Title:".bright_yellow(), story.title.bright_white());
        println!("   {} {}", "Author:".bright_yellow(),
                 story.author.as_deref().unwrap_or("Unknown").bright_white());
        println!("   {} {}", "Genre:".bright_yellow(), story.genre.join(", ").bright_white());
        println!("   {} {}", "Progress:".bright_yellow(),
                 format!("{:.1}%", story.progress_percentage()).bright_green());
        println!("   {} {}", "Word Count:".bright_yellow(),
                 format!("{} / {}",
                     story.metadata.current_word_count.to_string().bright_green(),
                     story.metadata.target_word_count.unwrap_or(0).to_string().bright_blue()
                 ));
        println!();

        // Chapter statistics
        let total_scenes = self.count_total_scenes(db, &chapters).await?;
        let completed_chapters = chapters.iter()
            .filter(|c| matches!(c.status, crate::story::ChapterStatus::Complete))
            .count();

        println!("{}", "📚 Structure Analysis".bright_white().bold());
        println!("   {} {}", "Chapters:".bright_yellow(), chapters.len().to_string().bright_green());
        println!("   {} {}", "Scenes:".bright_yellow(), total_scenes.to_string().bright_green());
        println!("   {} {}", "Completed Chapters:".bright_yellow(), completed_chapters.to_string().bright_blue());
        println!("   {} {}", "Average Scenes/Chapter:".bright_yellow(),
                 if chapters.is_empty() { "0".to_string() } else {
                     (total_scenes as f32 / chapters.len() as f32).to_string()
                 }.bright_green());
        println!();

        // Entity statistics
        let entity_stats = self.analyze_entity_distribution(&entities);
        println!("{}", "🎭 Entity Overview".bright_white().bold());
        println!("   {} {}", "Total Entities:".bright_yellow(), entities.len().to_string().bright_green());
        for (entity_type, count) in entity_stats {
            let color = match entity_type.as_str() {
                "Character" => "bright_green",
                "Technology" => "bright_blue",
                "Location" => "bright_yellow",
                "MacGuffin" => "bright_magenta",
                "Organization" => "bright_red",
                _ => "white",
            };
            println!("   {} {}",
                     format!("{}s:", entity_type).bright_yellow(),
                     count.to_string().color(color));
        }
        println!();

        // Relationship analysis
        let relationship_stats = self.analyze_relationships(&entities);
        println!("{}", "🕸️  Relationship Network".bright_white().bold());
        println!("   {} {}", "Total Relationships:".bright_yellow(),
                 relationship_stats.total_relationships.to_string().bright_green());
        println!("   {} {}", "Connected Characters:".bright_yellow(),
                 relationship_stats.connected_characters.to_string().bright_blue());
        println!("   {} {}", "Isolated Entities:".bright_yellow(),
                 relationship_stats.isolated_entities.to_string().bright_red());

        if let Some(ref most_connected) = relationship_stats.most_connected {
            println!("   {} {}", "Most Connected:".bright_yellow(), most_connected.bright_cyan());
        }

        Ok(())
    }

    /// Analyze and display entity relationships
    pub async fn show_relationship_analysis(&self, db: &StoryDatabase) -> Result<()> {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            return Ok(());
        }

        let story = &stories[0];
        let entities = db.list_entities(story.id, None).await?;

        println!("{}", "🕸️  Relationship Analysis".bright_cyan().bold());
        println!();

        // Character relationship matrix
        let characters: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type_name() == "Character")
            .collect();

        if !characters.is_empty() {
            println!("{}", "🎭 Character Relationships".bright_white().bold());
            for character in &characters {
                let relationships = character.relationships();
                if !relationships.is_empty() {
                    println!("   {} {}",
                             character.name().bright_green().bold(),
                             format!("({} connections)", relationships.len()).dimmed());

                    for rel in relationships {
                        // Find target character name
                        let target_name = entities.iter()
                            .find(|e| e.id == rel.target_id)
                            .map(|e| e.name())
                            .unwrap_or("Unknown");

                        println!("     {} {} - {}",
                                 "→".bright_blue(),
                                 target_name.bright_white(),
                                 rel.relationship_type.bright_yellow());

                        if !rel.description.is_empty() {
                            println!("       {}", rel.description.dimmed());
                        }
                    }
                    println!();
                }
            }
        }

        // Orphaned entities
        let orphaned: Vec<&Entity> = entities.iter()
            .filter(|e| e.relationships().is_empty() && e.appearances.is_empty())
            .collect();

        if !orphaned.is_empty() {
            println!("{}", "⚠️  Isolated Entities".bright_yellow().bold());
            for entity in orphaned {
                println!("   {} {} - {}",
                         entity.entity_type_name().bright_red(),
                         entity.name().bright_white(),
                         "No connections or appearances".dimmed());
            }
        }

        Ok(())
    }

    /// Show story structure analysis
    pub async fn show_story_structure(&self, db: &StoryDatabase) -> Result<()> {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            return Ok(());
        }

        let story = &stories[0];
        let chapters = db.list_chapters(story.id).await?;

        println!("{}", "🏗️  Story Structure Analysis".bright_cyan().bold());
        println!();

        // Three-act structure analysis
        let structure = self.analyze_story_structure(db, story, &chapters).await?;
        let three_act = structure.analyze_three_act_structure();

        println!("{}", "🎬 Three-Act Structure".bright_white().bold());
        println!("   {} {} (target: 25%)",
                 "Act I:".bright_green(),
                 format!("{:.1}%", three_act.act_1_actual * 100.0).bright_white());
        println!("   {} {} (target: 50%)",
                 "Act II:".bright_blue(),
                 format!("{:.1}%", three_act.act_2_actual * 100.0).bright_white());
        println!("   {} {} (target: 25%)",
                 "Act III:".bright_magenta(),
                 format!("{:.1}%", three_act.act_3_actual * 100.0).bright_white());

        let balance_score = three_act.balance_score * 100.0;
        let balance_color = if balance_score > 80.0 { "bright_green" }
                          else if balance_score > 60.0 { "bright_yellow" }
                          else { "bright_red" };
        println!("   {} {}",
                 "Balance Score:".bright_yellow(),
                 format!("{:.1}%", balance_score).color(balance_color));
        println!();

        // Chapter breakdown
        println!("{}", "📚 Chapter Breakdown".bright_white().bold());
        for (i, chapter) in chapters.iter().enumerate() {
            let scenes = db.list_scenes(chapter.id).await?;
            let status_color = match chapter.status {
                crate::story::ChapterStatus::Complete => "bright_green",
                crate::story::ChapterStatus::Editing => "bright_blue",
                crate::story::ChapterStatus::FirstDraft => "bright_yellow",
                crate::story::ChapterStatus::Drafting => "yellow",
                crate::story::ChapterStatus::Planned => "bright_red",
            };

            println!("   {} {} {} ({} scenes, {} words)",
                     format!("Ch {}:", i + 1).bright_cyan(),
                     chapter.title.bright_white(),
                     format!("[{:?}]", chapter.status).color(status_color),
                     scenes.len().to_string().dimmed(),
                     chapter.word_count.to_string().dimmed());
        }

        Ok(())
    }

    /// Generate comprehensive entity report
    pub async fn generate_entity_report(&self, db: &StoryDatabase) -> Result<()> {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            return Ok(());
        }

        let story = &stories[0];
        let analytics = db.generate_entity_analytics(story.id).await?;

        println!("{}", "📋 Entity Analytics Report".bright_cyan().bold());
        println!();

        println!("{}", "🎯 Summary".bright_white().bold());
        println!("   {} {}", "Total Entities:".bright_yellow(), analytics.total_entities.to_string().bright_green());
        println!("   {} {}", "Relationship Network Size:".bright_yellow(),
                 analytics.relationship_network_size.to_string().bright_blue());

        if let Some(ref character) = analytics.most_connected_character {
            println!("   {} {}", "Most Connected Character:".bright_yellow(), character.bright_cyan());
        }

        if let Some(ref location) = analytics.most_mentioned_location {
            println!("   {} {}", "Most Mentioned Location:".bright_yellow(), location.bright_green());
        }
        println!();

        // Entity type distribution
        println!("{}", "📊 Entity Distribution".bright_white().bold());
        for (entity_type, count) in &analytics.entity_counts {
            let percentage = (*count as f32 / analytics.total_entities as f32) * 100.0;
            println!("   {} {} ({})",
                     format!("{}:", entity_type).bright_yellow(),
                     count.to_string().bright_white(),
                     format!("{:.1}%", percentage).dimmed());
        }
        println!();

        // Orphaned entities warning
        if !analytics.orphaned_entities.is_empty() {
            println!("{}", "⚠️  Development Opportunities".bright_yellow().bold());
            println!("   {} entities have no relationships or appearances:",
                     analytics.orphaned_entities.len().to_string().bright_red());
            for entity in &analytics.orphaned_entities {
                println!("     • {}", entity.dimmed());
            }
            println!("   {} Consider developing these entities further.", "Tip:".bright_blue());
        }

        Ok(())
    }

    /// Show story timeline visualization
    pub async fn show_timeline(&self, db: &StoryDatabase) -> Result<()> {
        let stories = db.list_stories().await?;
        if stories.is_empty() {
            return Ok(());
        }

        let story = &stories[0];
        let chapters = db.list_chapters(story.id).await?;

        println!("{}", "📅 Story Timeline".bright_cyan().bold());
        println!();

        for (i, chapter) in chapters.iter().enumerate() {
            let scenes = db.list_scenes(chapter.id).await?;

            println!("{} {} {}",
                     format!("│ Chapter {}", i + 1).bright_blue(),
                     "─".repeat(50).dimmed(),
                     chapter.title.bright_white());

            for (j, scene) in scenes.iter().enumerate() {
                let connector = if j == scenes.len() - 1 { "└─" } else { "├─" };
                println!("{}   {} {}",
                         "│".bright_blue(),
                         connector.dimmed(),
                         scene.title.bright_green());

                if scene.tension_level > 7 {
                    println!("{}       {} High tension scene",
                             "│".bright_blue(),
                             "🔥".bright_red());
                }
            }

            if i < chapters.len() - 1 {
                println!("{}", "│".bright_blue());
            }
        }

        Ok(())
    }

    // Helper methods
    async fn count_total_scenes(&self, db: &StoryDatabase, chapters: &[Chapter]) -> Result<usize> {
        let mut total = 0;
        for chapter in chapters {
            let scenes = db.list_scenes(chapter.id).await?;
            total += scenes.len();
        }
        Ok(total)
    }

    fn analyze_entity_distribution(&self, entities: &[Entity]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entity in entities {
            let type_name = entity.entity_type_name().to_string();
            *counts.entry(type_name).or_insert(0) += 1;
        }
        counts
    }

    fn analyze_relationships(&self, entities: &[Entity]) -> RelationshipStats {
        let mut total_relationships = 0;
        let mut connected_characters = 0;
        let mut isolated_entities = 0;
        let mut connection_counts: HashMap<String, usize> = HashMap::new();

        for entity in entities {
            let relationships = entity.relationships();
            let has_appearances = !entity.appearances.is_empty();

            if relationships.is_empty() && !has_appearances {
                isolated_entities += 1;
            }

            if !relationships.is_empty() {
                total_relationships += relationships.len();
                if entity.entity_type_name() == "Character" {
                    connected_characters += 1;
                    connection_counts.insert(entity.name().to_string(), relationships.len());
                }
            }
        }

        let most_connected = connection_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(name, _)| name.clone());

        RelationshipStats {
            total_relationships,
            connected_characters,
            isolated_entities,
            most_connected,
        }
    }

    async fn analyze_story_structure(&self, db: &StoryDatabase, story: &Story, chapters: &[Chapter]) -> Result<StoryStructure> {
        let mut total_words = 0;
        let mut total_scenes = 0;

        // Calculate story metrics
        for chapter in chapters {
            total_words += chapter.word_count;
            let scenes = db.list_scenes(chapter.id).await?;
            total_scenes += scenes.len();
        }

        // Basic three-act division (simplified)
        let acts = vec![
            Act {
                name: "Act I".to_string(),
                chapters: chapters.iter().take(chapters.len() / 3).map(|c| c.id).collect(),
                word_count: total_words / 4, // Rough approximation
                percentage_of_story: 0.25,
                key_plot_points: vec![],
            },
            Act {
                name: "Act II".to_string(),
                chapters: chapters.iter().skip(chapters.len() / 3).take(chapters.len() / 2).map(|c| c.id).collect(),
                word_count: total_words / 2,
                percentage_of_story: 0.50,
                key_plot_points: vec![],
            },
            Act {
                name: "Act III".to_string(),
                chapters: chapters.iter().skip(2 * chapters.len() / 3).map(|c| c.id).collect(),
                word_count: total_words / 4,
                percentage_of_story: 0.25,
                key_plot_points: vec![],
            },
        ];

        let completion_status = if total_words > 0 {
            story.progress_percentage()
        } else {
            0.0
        };

        Ok(StoryStructure {
            acts,
            total_chapters: chapters.len() as u32,
            total_scenes: total_scenes as u32,
            total_words,
            completion_status,
        })
    }
}

#[derive(Debug)]
struct RelationshipStats {
    total_relationships: usize,
    connected_characters: usize,
    isolated_entities: usize,
    most_connected: Option<String>,
}