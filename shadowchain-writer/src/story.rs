use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use clap::Subcommand;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Story {
    pub id: Uuid,
    pub title: String,
    pub author: Option<String>,
    pub description: String,
    pub genre: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: StoryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryMetadata {
    pub target_word_count: Option<u32>,
    pub current_word_count: u32,
    pub target_audience: String,
    pub themes: Vec<String>,
    pub setting: String,
    pub time_period: String,
    pub pov: String, // First person, third person limited, etc.
    pub tense: String, // Past, present
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub id: Uuid,
    pub story_id: Uuid,
    pub title: String,
    pub description: String,
    pub order: u32,
    pub word_count: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub notes: String,
    pub status: ChapterStatus,
    pub act: Option<String>, // Act I, Act II, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChapterStatus {
    Planned,
    Drafting,
    FirstDraft,
    Editing,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub id: Uuid,
    pub chapter_id: Uuid,
    pub title: String,
    pub content: String,
    pub order: u32,
    pub word_count: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub notes: String,
    pub pov_character: Option<Uuid>, // Entity ID of POV character
    pub location: Option<Uuid>, // Entity ID of location
    pub tension_level: u8, // 1-10 scale
    pub scene_type: SceneType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneType {
    Action,
    Dialogue,
    Exposition,
    Transition,
    Climax,
    Resolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryArc {
    pub id: Uuid,
    pub story_id: Uuid,
    pub name: String,
    pub description: String,
    pub arc_type: ArcType,
    pub start_chapter: Uuid,
    pub end_chapter: Option<Uuid>,
    pub key_entities: Vec<Uuid>, // Characters involved in this arc
    pub resolution_status: ResolutionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArcType {
    Main,      // Primary story arc
    Character, // Character development arc
    Subplot,   // Secondary plot
    Mystery,   // Mystery to be solved
    Romance,   // Romantic subplot
    Revenge,   // Revenge arc
    Quest,     // Quest/mission arc
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStatus {
    Ongoing,
    Resolved,
    Abandoned,
    Cliffhanger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotPoint {
    pub id: Uuid,
    pub story_id: Uuid,
    pub name: String,
    pub description: String,
    pub plot_point_type: PlotPointType,
    pub chapter_id: Option<Uuid>,
    pub scene_id: Option<Uuid>,
    pub entities_involved: Vec<Uuid>,
    pub consequences: Vec<String>,
    pub foreshadowing: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotPointType {
    IncitingIncident,
    PlotPoint1,
    Midpoint,
    PlotPoint2,
    Climax,
    Resolution,
    TwistReveal,
    CharacterDeath,
    MajorChoice,
}

#[derive(Subcommand)]
pub enum StoryCommands {
    /// Chapter management
    Chapter {
        #[command(subcommand)]
        action: ChapterCommands,
    },
    /// Scene management
    Scene {
        #[command(subcommand)]
        action: SceneCommands,
    },
    /// Story arc management
    Arc {
        #[command(subcommand)]
        action: ArcCommands,
    },
    /// Plot point tracking
    Plot {
        #[command(subcommand)]
        action: PlotCommands,
    },
    /// Show story overview
    Overview,
    /// Story statistics
    Stats,
}

#[derive(Subcommand)]
pub enum ChapterCommands {
    /// Create a new chapter
    Create {
        title: String,
        #[arg(short, long)]
        description: Option<String>,
        #[arg(short, long)]
        act: Option<String>,
    },
    /// List all chapters
    List,
    /// Show chapter details
    Show {
        /// Chapter title or number
        chapter: String,
    },
    /// Edit chapter
    Edit {
        chapter: String,
    },
    /// Delete chapter
    Delete {
        chapter: String,
    },
}

#[derive(Subcommand)]
pub enum SceneCommands {
    /// Create a new scene
    Create {
        /// Chapter to add scene to
        chapter: String,
        /// Scene title
        title: String,
    },
    /// List scenes in a chapter
    List {
        /// Chapter title or number
        chapter: String,
    },
    /// Show scene content
    Show {
        chapter: String,
        scene: String,
    },
    /// Edit scene content
    Edit {
        chapter: String,
        scene: String,
    },
    /// Write scene content interactively
    Write {
        chapter: String,
        scene: String,
    },
}

#[derive(Subcommand)]
pub enum ArcCommands {
    /// Create story arc
    Create {
        name: String,
        #[arg(short, long)]
        arc_type: String,
    },
    /// List story arcs
    List,
    /// Show arc details
    Show {
        name: String,
    },
}

#[derive(Subcommand)]
pub enum PlotCommands {
    /// Add plot point
    Add {
        name: String,
        #[arg(short, long)]
        plot_type: String,
    },
    /// List plot points
    List,
    /// Show plot timeline
    Timeline,
}

impl Story {
    pub fn new(title: String, author: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title,
            author,
            description: String::new(),
            genre: vec!["Cyberpunk".to_string(), "Noir".to_string(), "Thriller".to_string()],
            created_at: now,
            updated_at: now,
            metadata: StoryMetadata {
                target_word_count: Some(80000), // Typical novel length
                current_word_count: 0,
                target_audience: "Adult".to_string(),
                themes: vec![
                    "Privacy vs. Surveillance".to_string(),
                    "Technology as weapon".to_string(),
                    "Identity and masks".to_string(),
                    "Obsession and revenge".to_string(),
                ],
                setting: "Near-future Berlin and global locations".to_string(),
                time_period: "2030s".to_string(),
                pov: "Third person limited".to_string(),
                tense: "Past tense".to_string(),
            },
        }
    }

    pub fn update_word_count(&mut self, new_count: u32) {
        self.metadata.current_word_count = new_count;
        self.updated_at = Utc::now();
    }

    pub fn progress_percentage(&self) -> f32 {
        if let Some(target) = self.metadata.target_word_count {
            if target > 0 {
                return (self.metadata.current_word_count as f32 / target as f32 * 100.0).min(100.0);
            }
        }
        0.0
    }
}

impl Chapter {
    pub fn new(story_id: Uuid, title: String, order: u32) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            story_id,
            title,
            description: String::new(),
            order,
            word_count: 0,
            created_at: now,
            updated_at: now,
            notes: String::new(),
            status: ChapterStatus::Planned,
            act: None,
        }
    }

    pub fn update_word_count(&mut self, scenes_word_count: u32) {
        self.word_count = scenes_word_count;
        self.updated_at = Utc::now();
    }

    pub fn set_status(&mut self, status: ChapterStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }
}

impl Scene {
    pub fn new(chapter_id: Uuid, title: String, content: String) -> Self {
        let word_count = count_words(&content);
        let now = Utc::now();

        Self {
            id: Uuid::new_v4(),
            chapter_id,
            title,
            content,
            order: 0, // Will be set when added to chapter
            word_count,
            created_at: now,
            updated_at: now,
            notes: String::new(),
            pov_character: None,
            location: None,
            tension_level: 5, // Default medium tension
            scene_type: SceneType::Exposition,
        }
    }

    pub fn update_content(&mut self, content: String) {
        self.content = content;
        self.word_count = count_words(&self.content);
        self.updated_at = Utc::now();
    }

    pub fn set_pov_character(&mut self, character_id: Uuid) {
        self.pov_character = Some(character_id);
        self.updated_at = Utc::now();
    }

    pub fn set_location(&mut self, location_id: Uuid) {
        self.location = Some(location_id);
        self.updated_at = Utc::now();
    }

    pub fn set_tension(&mut self, level: u8) {
        self.tension_level = level.min(10).max(1);
        self.updated_at = Utc::now();
    }
}

impl StoryArc {
    pub fn new(story_id: Uuid, name: String, arc_type: ArcType, start_chapter: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            story_id,
            name,
            description: String::new(),
            arc_type,
            start_chapter,
            end_chapter: None,
            key_entities: vec![],
            resolution_status: ResolutionStatus::Ongoing,
        }
    }

    pub fn add_character(&mut self, character_id: Uuid) {
        if !self.key_entities.contains(&character_id) {
            self.key_entities.push(character_id);
        }
    }

    pub fn resolve(&mut self, end_chapter: Uuid, status: ResolutionStatus) {
        self.end_chapter = Some(end_chapter);
        self.resolution_status = status;
    }
}

impl PlotPoint {
    pub fn new(story_id: Uuid, name: String, plot_point_type: PlotPointType) -> Self {
        Self {
            id: Uuid::new_v4(),
            story_id,
            name,
            description: String::new(),
            plot_point_type,
            chapter_id: None,
            scene_id: None,
            entities_involved: vec![],
            consequences: vec![],
            foreshadowing: vec![],
        }
    }

    pub fn attach_to_scene(&mut self, chapter_id: Uuid, scene_id: Uuid) {
        self.chapter_id = Some(chapter_id);
        self.scene_id = Some(scene_id);
    }

    pub fn add_consequence(&mut self, consequence: String) {
        self.consequences.push(consequence);
    }

    pub fn add_foreshadowing(&mut self, foreshadow: String) {
        self.foreshadowing.push(foreshadow);
    }
}

// Helper functions
fn count_words(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

/// Story structure analysis
#[derive(Debug, Clone)]
pub struct StoryStructure {
    pub acts: Vec<Act>,
    pub total_chapters: u32,
    pub total_scenes: u32,
    pub total_words: u32,
    pub completion_status: f32, // Percentage complete
}

#[derive(Debug, Clone)]
pub struct Act {
    pub name: String,
    pub chapters: Vec<Uuid>,
    pub word_count: u32,
    pub percentage_of_story: f32,
    pub key_plot_points: Vec<Uuid>,
}

impl StoryStructure {
    pub fn analyze_three_act_structure(&self) -> ThreeActAnalysis {
        let act_1_target = 0.25; // 25% of story
        let act_2_target = 0.50; // 50% of story
        let act_3_target = 0.25; // 25% of story

        ThreeActAnalysis {
            act_1_actual: self.acts.get(0).map(|a| a.percentage_of_story).unwrap_or(0.0),
            act_2_actual: self.acts.get(1).map(|a| a.percentage_of_story).unwrap_or(0.0),
            act_3_actual: self.acts.get(2).map(|a| a.percentage_of_story).unwrap_or(0.0),
            act_1_target,
            act_2_target,
            act_3_target,
            balance_score: self.calculate_balance_score(),
        }
    }

    fn calculate_balance_score(&self) -> f32 {
        // Calculate how well-balanced the three-act structure is
        if self.acts.len() != 3 {
            return 0.0;
        }

        let ideal = [0.25, 0.50, 0.25];
        let actual: Vec<f32> = self.acts.iter().map(|a| a.percentage_of_story).collect();

        let total_deviation: f32 = ideal.iter().zip(actual.iter())
            .map(|(i, a)| (i - a).abs())
            .sum();

        // Perfect score is 1.0, worst score approaches 0.0
        (1.0 - total_deviation).max(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct ThreeActAnalysis {
    pub act_1_actual: f32,
    pub act_2_actual: f32,
    pub act_3_actual: f32,
    pub act_1_target: f32,
    pub act_2_target: f32,
    pub act_3_target: f32,
    pub balance_score: f32, // 0.0 to 1.0, how well balanced
}

/// Pacing analysis
#[derive(Debug, Clone)]
pub struct PacingAnalysis {
    pub tension_curve: Vec<TensionPoint>,
    pub pacing_score: f32,
    pub slow_sections: Vec<SceneRange>,
    pub rushed_sections: Vec<SceneRange>,
    pub climax_positioning: f32, // Where in story (0.0 to 1.0)
}

#[derive(Debug, Clone)]
pub struct TensionPoint {
    pub chapter_id: Uuid,
    pub scene_id: Uuid,
    pub tension_level: u8,
    pub story_position: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone)]
pub struct SceneRange {
    pub start_scene: Uuid,
    pub end_scene: Uuid,
    pub severity: f32, // How problematic (0.0 to 1.0)
}

/// Character development tracking
#[derive(Debug, Clone)]
pub struct CharacterDevelopment {
    pub character_id: Uuid,
    pub character_name: String,
    pub development_arc: Vec<DevelopmentPoint>,
    pub growth_score: f32,
    pub screen_time: u32, // Word count in scenes
    pub importance_rank: u32,
}

#[derive(Debug, Clone)]
pub struct DevelopmentPoint {
    pub scene_id: Uuid,
    pub development_type: DevelopmentType,
    pub description: String,
    pub story_position: f32,
}

#[derive(Debug, Clone)]
pub enum DevelopmentType {
    Introduction,
    Growth,
    Setback,
    Revelation,
    Transformation,
    Climax,
    Resolution,
}