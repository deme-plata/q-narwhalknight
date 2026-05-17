use anyhow::Result;
use colored::*;
use std::collections::HashMap;
use uuid::Uuid;

use crate::database::StoryDatabase;
use crate::entities::{Entity, EntityType};

/// AI-assisted writing tools and suggestions
pub struct AIAssistant {
    creativity_engine: CreativityEngine,
    style_analyzer: StyleAnalyzer,
    plot_generator: PlotGenerator,
}

impl AIAssistant {
    pub fn new() -> Self {
        Self {
            creativity_engine: CreativityEngine::new(),
            style_analyzer: StyleAnalyzer::new(),
            plot_generator: PlotGenerator::new(),
        }
    }

    /// Generate character description and traits
    pub async fn generate_character(&self, db: &StoryDatabase, name: &str) -> Result<()> {
        println!("{}", format!("🤖 AI Character Generator: {}", name).bright_cyan().bold());
        println!();

        let character_suggestions = self.creativity_engine.generate_cyberpunk_character(name);

        println!("{}", "Generated Character Profile:".bright_white().bold());
        println!("   {} {}", "Name:".bright_yellow(), character_suggestions.name.bright_green());
        println!("   {} {}", "Archetype:".bright_yellow(), character_suggestions.archetype.bright_blue());
        println!("   {} {}", "Background:".bright_yellow(), character_suggestions.background.bright_white());
        println!();

        println!("{}", "Suggested Traits:".bright_white().bold());
        for character_trait in &character_suggestions.traits {
            println!("   • {}", character_trait.bright_green());
        }
        println!();

        println!("{}", "Potential Motivations:".bright_white().bold());
        for motivation in &character_suggestions.motivations {
            println!("   • {}", motivation.bright_blue());
        }
        println!();

        println!("{}", "Cyberpunk Elements:".bright_white().bold());
        for element in &character_suggestions.cyberpunk_elements {
            println!("   • {}", element.bright_magenta());
        }
        println!();

        println!("{}", "💡 Usage Tip:".bright_cyan());
        println!("   Use 'shadowchain entity create --entity-type character {}' to add this character", name);

        Ok(())
    }

    /// Suggest plot developments
    pub async fn suggest_plot_developments(&self, db: &StoryDatabase) -> Result<()> {
        println!("{}", "🧠 AI Plot Development Suggestions".bright_cyan().bold());
        println!();

        let stories = db.list_stories().await?;
        if stories.is_empty() {
            println!("{}", "No stories found. Initialize a story first.".red());
            return Ok(());
        }

        let story = &stories[0];
        let entities = db.list_entities(story.id, None).await?;
        let chapters = db.list_chapters(story.id).await?;

        let plot_suggestions = self.plot_generator.analyze_story_and_suggest(&entities, &chapters);

        println!("{}", "🎬 Plot Development Opportunities:".bright_white().bold());
        for suggestion in &plot_suggestions.development_ideas {
            println!("   {} {}", "•".bright_green(), suggestion.bright_white());
        }
        println!();

        println!("{}", "⚡ Tension Building Suggestions:".bright_white().bold());
        for tension in &plot_suggestions.tension_points {
            println!("   {} {}", "•".bright_red(), tension.bright_white());
        }
        println!();

        println!("{}", "🔗 Relationship Developments:".bright_white().bold());
        for relationship in &plot_suggestions.relationship_arcs {
            println!("   {} {}", "•".bright_blue(), relationship.bright_white());
        }
        println!();

        if !plot_suggestions.underutilized_entities.is_empty() {
            println!("{}", "⚠️  Underutilized Entities:".bright_yellow().bold());
            for entity in &plot_suggestions.underutilized_entities {
                println!("   {} {} - Consider expanding their role", "•".yellow(), entity.bright_white());
            }
            println!();
        }

        Ok(())
    }

    /// Analyze writing style
    pub async fn analyze_writing_style(&self, db: &StoryDatabase) -> Result<()> {
        println!("{}", "📝 Writing Style Analysis".bright_cyan().bold());
        println!();

        let stories = db.list_stories().await?;
        if stories.is_empty() {
            println!("{}", "No stories found. Write some content first.".red());
            return Ok(());
        }

        let story = &stories[0];
        let chapters = db.list_chapters(story.id).await?;

        let mut total_content = String::new();
        for chapter in &chapters {
            let scenes = db.list_scenes(chapter.id).await?;
            for scene in scenes {
                total_content.push_str(&scene.content);
                total_content.push(' ');
            }
        }

        if total_content.trim().is_empty() {
            println!("{}", "No content found to analyze. Write some scenes first.".yellow());
            return Ok(());
        }

        let style_analysis = self.style_analyzer.analyze_text(&total_content);

        println!("{}", "📊 Style Metrics:".bright_white().bold());
        println!("   {} {}", "Avg Sentence Length:".bright_yellow(),
                 format!("{:.1} words", style_analysis.avg_sentence_length).bright_green());
        println!("   {} {}", "Vocabulary Richness:".bright_yellow(),
                 format!("{:.1}%", style_analysis.vocabulary_richness * 100.0).bright_blue());
        println!("   {} {}", "Reading Level:".bright_yellow(), style_analysis.reading_level.bright_white());
        println!("   {} {}", "Tone:".bright_yellow(), style_analysis.dominant_tone.bright_magenta());
        println!();

        println!("{}", "🎭 Style Characteristics:".bright_white().bold());
        for characteristic in &style_analysis.style_notes {
            println!("   • {}", characteristic.bright_cyan());
        }
        println!();

        println!("{}", "📈 Improvement Suggestions:".bright_white().bold());
        for suggestion in &style_analysis.suggestions {
            println!("   • {}", suggestion.bright_green());
        }

        Ok(())
    }

    /// Generate technology descriptions
    pub async fn generate_technology(&self, db: &StoryDatabase, name: &str) -> Result<()> {
        println!("{}", format!("⚡ AI Technology Generator: {}", name).bright_cyan().bold());
        println!();

        let tech_suggestions = self.creativity_engine.generate_cyberpunk_technology(name);

        println!("{}", "Generated Technology Profile:".bright_white().bold());
        println!("   {} {}", "Name:".bright_yellow(), tech_suggestions.name.bright_green());
        println!("   {} {}", "Category:".bright_yellow(), tech_suggestions.category.bright_blue());
        println!("   {} {}", "Description:".bright_yellow(), tech_suggestions.description.bright_white());
        println!();

        println!("{}", "Technical Specifications:".bright_white().bold());
        for spec in &tech_suggestions.technical_details {
            println!("   • {}", spec.bright_green());
        }
        println!();

        println!("{}", "Story Integration Ideas:".bright_white().bold());
        for idea in &tech_suggestions.story_hooks {
            println!("   • {}", idea.bright_blue());
        }
        println!();

        println!("{}", "Cyberpunk Implications:".bright_white().bold());
        for implication in &tech_suggestions.cyberpunk_implications {
            println!("   • {}", implication.bright_magenta());
        }
        println!();

        println!("{}", "💡 Usage Tip:".bright_cyan());
        println!("   Use 'shadowchain entity create --entity-type tech {}' to add this technology", name);

        Ok(())
    }
}

/// Creative content generation engine
struct CreativityEngine {
    cyberpunk_traits: Vec<String>,
    cyberpunk_archetypes: Vec<String>,
    tech_categories: Vec<String>,
}

impl CreativityEngine {
    fn new() -> Self {
        Self {
            cyberpunk_traits: vec![
                "tech-savvy".to_string(),
                "paranoid".to_string(),
                "augmented".to_string(),
                "street-smart".to_string(),
                "jaded".to_string(),
                "networked".to_string(),
                "code-slinger".to_string(),
                "ghost-walker".to_string(),
                "neon-shadow".to_string(),
                "data-miner".to_string(),
            ],
            cyberpunk_archetypes: vec![
                "Hacker".to_string(),
                "Corporate Agent".to_string(),
                "Street Samurai".to_string(),
                "Fixer".to_string(),
                "Netrunner".to_string(),
                "Tech Specialist".to_string(),
                "Data Broker".to_string(),
                "Smuggler".to_string(),
                "Investigator".to_string(),
                "Ghost in the Machine".to_string(),
            ],
            tech_categories: vec![
                "Cryptographic Protocol".to_string(),
                "Neural Interface".to_string(),
                "Blockchain Technology".to_string(),
                "Surveillance System".to_string(),
                "Anonymity Tool".to_string(),
                "Data Mining Platform".to_string(),
                "Quantum Computer".to_string(),
                "AI Assistant".to_string(),
                "Biometric Scanner".to_string(),
                "Steganography Tool".to_string(),
            ],
        }
    }

    fn generate_cyberpunk_character(&self, name: &str) -> CharacterSuggestion {
        // Simple deterministic generation based on name hash
        let hash = self.simple_hash(name);
        let archetype_idx = hash % self.cyberpunk_archetypes.len();
        let trait_count = 3 + (hash % 3);

        CharacterSuggestion {
            name: name.to_string(),
            archetype: self.cyberpunk_archetypes[archetype_idx].clone(),
            background: format!("A {} operating in the shadows of the digital underworld",
                              self.cyberpunk_archetypes[archetype_idx].to_lowercase()),
            traits: (0..trait_count).map(|i| {
                let idx = (hash + i * 7) % self.cyberpunk_traits.len();
                self.cyberpunk_traits[idx].clone()
            }).collect(),
            motivations: vec![
                "Seeking freedom from corporate control".to_string(),
                "Hunting for the truth behind a conspiracy".to_string(),
                "Protecting someone they care about".to_string(),
                "Pursuing revenge against a powerful enemy".to_string(),
            ],
            cyberpunk_elements: vec![
                "Augmented with cutting-edge implants".to_string(),
                "Has access to underground networks".to_string(),
                "Skilled in digital warfare and encryption".to_string(),
                "Haunted by past corporate betrayals".to_string(),
            ],
        }
    }

    fn generate_cyberpunk_technology(&self, name: &str) -> TechnologySuggestion {
        let hash = self.simple_hash(name);
        let category_idx = hash % self.tech_categories.len();

        TechnologySuggestion {
            name: name.to_string(),
            category: self.tech_categories[category_idx].clone(),
            description: format!("Advanced {} designed for covert operations",
                               self.tech_categories[category_idx].to_lowercase()),
            technical_details: vec![
                "Quantum-encrypted communications".to_string(),
                "Zero-knowledge proof verification".to_string(),
                "Distributed hash table storage".to_string(),
                "Post-quantum cryptographic signatures".to_string(),
            ],
            story_hooks: vec![
                "Could be the key to exposing a conspiracy".to_string(),
                "Might be used to track the protagonists".to_string(),
                "Could provide access to restricted networks".to_string(),
                "May contain hidden backdoors".to_string(),
            ],
            cyberpunk_implications: vec![
                "Challenges traditional notions of privacy".to_string(),
                "Enables new forms of digital resistance".to_string(),
                "Could shift the balance of power".to_string(),
                "Represents the double-edged nature of technology".to_string(),
            ],
        }
    }

    fn simple_hash(&self, s: &str) -> usize {
        s.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize))
    }
}

/// Writing style analysis engine
struct StyleAnalyzer {
    common_words: Vec<String>,
}

impl StyleAnalyzer {
    fn new() -> Self {
        Self {
            common_words: vec![
                "the".to_string(), "and".to_string(), "of".to_string(), "to".to_string(),
                "a".to_string(), "in".to_string(), "is".to_string(), "it".to_string(),
                "you".to_string(), "that".to_string(), "he".to_string(), "was".to_string(),
                "for".to_string(), "on".to_string(), "are".to_string(), "as".to_string(),
            ],
        }
    }

    fn analyze_text(&self, text: &str) -> StyleAnalysis {
        let sentences = self.split_sentences(text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();

        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            words.len() as f32 / sentences.len() as f32
        };

        let vocabulary_richness = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f32 / words.len() as f32
        };

        StyleAnalysis {
            avg_sentence_length,
            vocabulary_richness,
            reading_level: self.estimate_reading_level(avg_sentence_length, vocabulary_richness),
            dominant_tone: self.analyze_tone(text),
            style_notes: self.generate_style_notes(avg_sentence_length, vocabulary_richness),
            suggestions: self.generate_suggestions(avg_sentence_length, vocabulary_richness),
        }
    }

    fn split_sentences(&self, text: &str) -> Vec<String> {
        text.split(&['.', '!', '?'][..])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn estimate_reading_level(&self, avg_sentence_length: f32, vocabulary_richness: f32) -> String {
        match (avg_sentence_length, vocabulary_richness) {
            (s, v) if s < 10.0 && v > 0.7 => "Elementary",
            (s, v) if s < 15.0 && v > 0.5 => "Middle School",
            (s, v) if s < 20.0 && v > 0.3 => "High School",
            _ => "College Level",
        }.to_string()
    }

    fn analyze_tone(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        if text_lower.contains("dark") || text_lower.contains("shadow") || text_lower.contains("grim") {
            "Noir/Dark".to_string()
        } else if text_lower.contains("tech") || text_lower.contains("cyber") || text_lower.contains("digital") {
            "Technical/Cyberpunk".to_string()
        } else if text_lower.contains("fast") || text_lower.contains("quick") || text_lower.contains("sudden") {
            "Action-packed".to_string()
        } else {
            "Neutral".to_string()
        }
    }

    fn generate_style_notes(&self, avg_sentence_length: f32, vocabulary_richness: f32) -> Vec<String> {
        let mut notes = Vec::new();

        if avg_sentence_length > 20.0 {
            notes.push("Uses complex, lengthy sentences".to_string());
        } else if avg_sentence_length < 10.0 {
            notes.push("Employs short, punchy sentences".to_string());
        }

        if vocabulary_richness > 0.6 {
            notes.push("Rich and varied vocabulary".to_string());
        } else if vocabulary_richness < 0.3 {
            notes.push("Simple, direct word choices".to_string());
        }

        notes.push("Cyberpunk noir aesthetic".to_string());
        notes.push("Technical terminology integration".to_string());

        notes
    }

    fn generate_suggestions(&self, avg_sentence_length: f32, vocabulary_richness: f32) -> Vec<String> {
        let mut suggestions = Vec::new();

        if avg_sentence_length > 25.0 {
            suggestions.push("Consider breaking up some longer sentences for better flow".to_string());
        } else if avg_sentence_length < 8.0 {
            suggestions.push("Try combining some short sentences for variety".to_string());
        }

        if vocabulary_richness < 0.4 {
            suggestions.push("Experiment with more diverse vocabulary".to_string());
        }

        suggestions.push("Maintain the cyberpunk atmosphere with technical details".to_string());
        suggestions.push("Use sensory details to enhance the noir mood".to_string());

        suggestions
    }
}

/// Plot development suggestion engine
struct PlotGenerator;

impl PlotGenerator {
    fn new() -> Self {
        Self
    }

    fn analyze_story_and_suggest(&self, entities: &[Entity], chapters: &[crate::story::Chapter]) -> PlotSuggestions {
        let underutilized = self.find_underutilized_entities(entities);

        PlotSuggestions {
            development_ideas: vec![
                "Introduce a double agent within the organization".to_string(),
                "Reveal a hidden connection between two seemingly unrelated characters".to_string(),
                "Create a time pressure element with a looming deadline".to_string(),
                "Add a technological breakthrough that changes the game".to_string(),
                "Introduce a moral dilemma that tests character loyalties".to_string(),
            ],
            tension_points: vec![
                "A trusted ally betrays the protagonist".to_string(),
                "The antagonist captures someone important".to_string(),
                "A critical system failure at the worst moment".to_string(),
                "Discovery of a larger conspiracy".to_string(),
                "A character must choose between personal and greater good".to_string(),
            ],
            relationship_arcs: vec![
                "Develop the mentor-student dynamic".to_string(),
                "Explore the complexity of the enemy relationship".to_string(),
                "Show character growth through adversity".to_string(),
                "Build romantic tension amid the chaos".to_string(),
                "Test loyalty through difficult circumstances".to_string(),
            ],
            underutilized_entities: underutilized,
        }
    }

    fn find_underutilized_entities(&self, entities: &[Entity]) -> Vec<String> {
        entities.iter()
            .filter(|e| e.relationships().is_empty() && e.appearances.is_empty())
            .map(|e| e.name().to_string())
            .collect()
    }
}

// Data structures for AI suggestions
#[derive(Debug)]
struct CharacterSuggestion {
    name: String,
    archetype: String,
    background: String,
    traits: Vec<String>,
    motivations: Vec<String>,
    cyberpunk_elements: Vec<String>,
}

#[derive(Debug)]
struct TechnologySuggestion {
    name: String,
    category: String,
    description: String,
    technical_details: Vec<String>,
    story_hooks: Vec<String>,
    cyberpunk_implications: Vec<String>,
}

#[derive(Debug)]
struct StyleAnalysis {
    avg_sentence_length: f32,
    vocabulary_richness: f32,
    reading_level: String,
    dominant_tone: String,
    style_notes: Vec<String>,
    suggestions: Vec<String>,
}

#[derive(Debug)]
struct PlotSuggestions {
    development_ideas: Vec<String>,
    tension_points: Vec<String>,
    relationship_arcs: Vec<String>,
    underutilized_entities: Vec<String>,
}