use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Core entity types in the story universe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Character {
        name: String,
        description: String,
        traits: Vec<String>,
        relationships: Vec<Relationship>,
    },
    Technology {
        name: String,
        description: String,
        technical_details: Vec<String>,
    },
    Location {
        name: String,
        description: String,
        atmosphere: String,
        significance: Vec<String>,
    },
    MacGuffin {
        name: String,
        description: String,
        importance: String,
        location: String,
    },
    Organization {
        name: String,
        description: String,
        members: Vec<Uuid>,
        agenda: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub target_id: Uuid,
    pub relationship_type: String, // "enemy", "ally", "mentor", "lover", etc.
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub story_id: Uuid,
    pub entity_type: EntityType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub notes: String,
    pub appearances: Vec<SceneAppearance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAppearance {
    pub chapter_id: Uuid,
    pub scene_id: Uuid,
    pub importance: AppearanceImportance,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AppearanceImportance {
    Main,      // Protagonist/key player in scene
    Supporting,// Important but not central
    Mentioned, // Referenced but not present
    Cameo,     // Brief appearance
}

impl EntityType {
    pub fn name(&self) -> &str {
        match self {
            EntityType::Character { name, .. } => name,
            EntityType::Technology { name, .. } => name,
            EntityType::Location { name, .. } => name,
            EntityType::MacGuffin { name, .. } => name,
            EntityType::Organization { name, .. } => name,
        }
    }
}

impl Entity {
    pub fn name(&self) -> &str {
        match &self.entity_type {
            EntityType::Character { name, .. } => name,
            EntityType::Technology { name, .. } => name,
            EntityType::Location { name, .. } => name,
            EntityType::MacGuffin { name, .. } => name,
            EntityType::Organization { name, .. } => name,
        }
    }

    pub fn description(&self) -> &str {
        match &self.entity_type {
            EntityType::Character { description, .. } => description,
            EntityType::Technology { description, .. } => description,
            EntityType::Location { description, .. } => description,
            EntityType::MacGuffin { description, .. } => description,
            EntityType::Organization { description, .. } => description,
        }
    }

    pub fn entity_type_name(&self) -> &'static str {
        match &self.entity_type {
            EntityType::Character { .. } => "Character",
            EntityType::Technology { .. } => "Technology",
            EntityType::Location { .. } => "Location",
            EntityType::MacGuffin { .. } => "MacGuffin",
            EntityType::Organization { .. } => "Organization",
        }
    }

    /// Get relationships for characters
    pub fn relationships(&self) -> Vec<&Relationship> {
        match &self.entity_type {
            EntityType::Character { relationships, .. } => relationships.iter().collect(),
            _ => vec![],
        }
    }

    /// Add a relationship (for characters)
    pub fn add_relationship(&mut self, relationship: Relationship) {
        if let EntityType::Character { relationships, .. } = &mut self.entity_type {
            relationships.push(relationship);
        }
    }

    /// Add scene appearance
    pub fn add_appearance(&mut self, appearance: SceneAppearance) {
        self.appearances.push(appearance);
        self.updated_at = Utc::now();
    }

    /// Get technical details (for technology entities)
    pub fn technical_details(&self) -> Vec<&String> {
        match &self.entity_type {
            EntityType::Technology { technical_details, .. } => technical_details.iter().collect(),
            _ => vec![],
        }
    }
}

/// Story-wide entity analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAnalytics {
    pub total_entities: usize,
    pub entity_counts: HashMap<String, usize>,
    pub most_connected_character: Option<String>,
    pub most_mentioned_location: Option<String>,
    pub relationship_network_size: usize,
    pub orphaned_entities: Vec<String>, // Entities with no connections
}

/// Generate relationship graph data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub importance_score: f64, // Based on appearances and connections
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub strength: f64, // How often they appear together
}

/// Entity search and filtering
#[derive(Debug, Clone)]
pub struct EntityFilter {
    pub entity_types: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub name_contains: Option<String>,
    pub has_relationships: Option<bool>,
    pub appears_in_chapter: Option<Uuid>,
}

impl EntityFilter {
    pub fn new() -> Self {
        Self {
            entity_types: None,
            tags: None,
            name_contains: None,
            has_relationships: None,
            appears_in_chapter: None,
        }
    }

    pub fn with_type(mut self, entity_type: &str) -> Self {
        self.entity_types = Some(vec![entity_type.to_string()]);
        self
    }

    pub fn with_name_containing(mut self, name: &str) -> Self {
        self.name_contains = Some(name.to_string());
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        match &mut self.tags {
            Some(tags) => tags.push(tag.to_string()),
            None => self.tags = Some(vec![tag.to_string()]),
        }
        self
    }
}

/// Cyberpunk-specific entity templates
pub mod templates {
    use super::*;

    pub fn create_hacker_character(name: String, specialization: String) -> EntityType {
        EntityType::Character {
            name,
            description: format!("Elite hacker specializing in {}", specialization),
            traits: vec![
                "tech-savvy".to_string(),
                "paranoid".to_string(),
                "night-owl".to_string(),
            ],
            relationships: vec![],
        }
    }

    pub fn create_crypto_tech(name: String, algorithm: String) -> EntityType {
        EntityType::Technology {
            name,
            description: format!("Cryptographic technology based on {}", algorithm),
            technical_details: vec![
                format!("Algorithm: {}", algorithm),
                "Quantum-resistant".to_string(),
                "Zero-knowledge proofs".to_string(),
            ],
        }
    }

    pub fn create_underground_location(name: String, city: String) -> EntityType {
        EntityType::Location {
            name,
            description: format!("Underground hideout in {}", city),
            atmosphere: "Dark, neon-lit, tech-heavy".to_string(),
            significance: vec!["Safe house".to_string(), "Meeting point".to_string()],
        }
    }

    pub fn create_shadow_organization(name: String, purpose: String) -> EntityType {
        EntityType::Organization {
            name,
            description: format!("Covert organization focused on {}", purpose),
            members: vec![],
            agenda: purpose,
        }
    }
}