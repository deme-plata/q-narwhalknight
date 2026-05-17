use anyhow::{Result, Context};
use rocksdb::{DB, IteratorMode, Options};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::entities::{Entity, EntityType, EntityFilter, RelationshipGraph, EntityAnalytics};
use crate::story::{Story, Chapter, Scene};

/// RocksDB-backed story database
pub struct StoryDatabase {
    db: DB,
    path: String,
}

impl StoryDatabase {
    pub async fn new(path: &str) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let db = DB::open(&opts, path)
            .context("Failed to open RocksDB database")?;

        Ok(Self {
            db,
            path: path.to_string(),
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    // Story management
    pub async fn create_story(&mut self, title: &str, author: Option<&str>) -> Result<Uuid> {
        let story = Story::new(title.to_string(), author.map(|s| s.to_string()));
        let key = format!("story:{}", story.id);
        let value = serde_json::to_vec(&story)?;

        self.db.put(&key, &value)?;

        // Update story index
        self.update_story_index(&story).await?;

        Ok(story.id)
    }

    pub async fn get_story(&self, story_id: Uuid) -> Result<Option<Story>> {
        let key = format!("story:{}", story_id);
        match self.db.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    pub async fn list_stories(&self) -> Result<Vec<Story>> {
        let mut stories = Vec::new();
        let iter = self.db.iterator(IteratorMode::From(b"story:", rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if key_str.starts_with("story:") && !key_str.ends_with(":index") {
                let story: Story = serde_json::from_slice(&value)?;
                stories.push(story);
            }
        }

        // Sort by creation date (most recent first)
        stories.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(stories)
    }

    // Entity management
    pub async fn create_entity(&mut self, story_id: Uuid, entity_type: EntityType) -> Result<Uuid> {
        let entity = Entity {
            id: Uuid::new_v4(),
            story_id,
            entity_type,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            tags: vec![],
            notes: String::new(),
            appearances: vec![],
        };

        let key = format!("entity:{}:{}", story_id, entity.id);
        let value = serde_json::to_vec(&entity)?;

        self.db.put(&key, &value)?;

        // Update entity index
        self.update_entity_index(&entity).await?;

        Ok(entity.id)
    }

    pub async fn get_entity(&self, story_id: Uuid, entity_id: Uuid) -> Result<Option<Entity>> {
        let key = format!("entity:{}:{}", story_id, entity_id);
        match self.db.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    pub async fn update_entity(&mut self, entity: &Entity) -> Result<()> {
        let key = format!("entity:{}:{}", entity.story_id, entity.id);
        let value = serde_json::to_vec(entity)?;

        self.db.put(&key, &value)?;

        // Update index
        self.update_entity_index(entity).await?;

        Ok(())
    }

    pub async fn list_entities(&self, story_id: Uuid, filter: Option<EntityFilter>) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        let prefix = format!("entity:{}:", story_id);
        let iter = self.db.iterator(IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }
            if key_str.ends_with(":index") {
                continue;
            }

            let entity: Entity = serde_json::from_slice(&value)?;

            // Apply filter if provided
            if let Some(ref filter) = filter {
                if !self.entity_matches_filter(&entity, filter) {
                    continue;
                }
            }

            entities.push(entity);
        }

        // Sort by name
        entities.sort_by(|a, b| a.name().cmp(b.name()));
        Ok(entities)
    }

    pub async fn search_entities(&self, story_id: Uuid, query: &str) -> Result<Vec<Entity>> {
        let entities = self.list_entities(story_id, None).await?;
        let mut matches = Vec::new();

        let query = query.to_lowercase();

        for entity in entities {
            // Fuzzy search on name and description
            if entity.name().to_lowercase().contains(&query) ||
               entity.description().to_lowercase().contains(&query) {
                matches.push(entity);
            }
        }

        Ok(matches)
    }

    pub async fn delete_entity(&mut self, story_id: Uuid, entity_id: Uuid) -> Result<()> {
        let key = format!("entity:{}:{}", story_id, entity_id);
        self.db.delete(&key)?;

        // Remove from index
        let index_key = format!("entity:{}:{}:index", story_id, entity_id);
        self.db.delete(&index_key)?;

        Ok(())
    }

    // Chapter and scene management
    pub async fn create_chapter(&mut self, story_id: Uuid, title: String, order: u32) -> Result<Uuid> {
        let chapter = Chapter::new(story_id, title, order);
        let key = format!("chapter:{}:{}", story_id, chapter.id);
        let value = serde_json::to_vec(&chapter)?;

        self.db.put(&key, &value)?;
        Ok(chapter.id)
    }

    pub async fn get_chapter(&self, story_id: Uuid, chapter_id: Uuid) -> Result<Option<Chapter>> {
        let key = format!("chapter:{}:{}", story_id, chapter_id);
        match self.db.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    pub async fn list_chapters(&self, story_id: Uuid) -> Result<Vec<Chapter>> {
        let mut chapters = Vec::new();
        let prefix = format!("chapter:{}:", story_id);
        let iter = self.db.iterator(IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }

            let chapter: Chapter = serde_json::from_slice(&value)?;
            chapters.push(chapter);
        }

        // Sort by order
        chapters.sort_by_key(|c| c.order);
        Ok(chapters)
    }

    pub async fn create_scene(&mut self, chapter_id: Uuid, title: String, content: String) -> Result<Uuid> {
        let scene = Scene::new(chapter_id, title, content);
        let key = format!("scene:{}:{}", chapter_id, scene.id);
        let value = serde_json::to_vec(&scene)?;

        self.db.put(&key, &value)?;
        Ok(scene.id)
    }

    pub async fn get_scene(&self, chapter_id: Uuid, scene_id: Uuid) -> Result<Option<Scene>> {
        let key = format!("scene:{}:{}", chapter_id, scene_id);
        match self.db.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    pub async fn list_scenes(&self, chapter_id: Uuid) -> Result<Vec<Scene>> {
        let mut scenes = Vec::new();
        let prefix = format!("scene:{}:", chapter_id);
        let iter = self.db.iterator(IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if !key_str.starts_with(&prefix) {
                break;
            }

            let scene: Scene = serde_json::from_slice(&value)?;
            scenes.push(scene);
        }

        // Sort by order
        scenes.sort_by_key(|s| s.order);
        Ok(scenes)
    }

    // Analytics and visualization
    pub async fn generate_entity_analytics(&self, story_id: Uuid) -> Result<EntityAnalytics> {
        let entities = self.list_entities(story_id, None).await?;

        let mut entity_counts = std::collections::HashMap::new();
        let mut total_relationships = 0;
        let mut character_connection_counts = std::collections::HashMap::new();

        for entity in &entities {
            let type_name = entity.entity_type_name();
            *entity_counts.entry(type_name.to_string()).or_insert(0) += 1;

            // Count relationships for characters
            let relationships = entity.relationships();
            total_relationships += relationships.len();

            if !relationships.is_empty() {
                character_connection_counts.insert(entity.name().to_string(), relationships.len());
            }
        }

        let most_connected_character = character_connection_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(name, _)| name.clone());

        // Find orphaned entities (no relationships, no appearances)
        let orphaned_entities: Vec<String> = entities
            .iter()
            .filter(|e| e.relationships().is_empty() && e.appearances.is_empty())
            .map(|e| e.name().to_string())
            .collect();

        Ok(EntityAnalytics {
            total_entities: entities.len(),
            entity_counts,
            most_connected_character,
            most_mentioned_location: None, // TODO: Implement based on appearances
            relationship_network_size: total_relationships,
            orphaned_entities,
        })
    }

    // Private helper methods
    async fn update_story_index(&self, story: &Story) -> Result<()> {
        let index_key = format!("story:{}:index", story.id);
        let index_data = serde_json::json!({
            "title": story.title,
            "author": story.author,
            "created_at": story.created_at,
        });
        let value = serde_json::to_vec(&index_data)?;
        self.db.put(&index_key, &value)?;
        Ok(())
    }

    async fn update_entity_index(&self, entity: &Entity) -> Result<()> {
        let index_key = format!("entity:{}:{}:index", entity.story_id, entity.id);
        let index_data = serde_json::json!({
            "name": entity.name(),
            "type": entity.entity_type_name(),
            "description": entity.description(),
            "tags": entity.tags,
            "updated_at": entity.updated_at,
        });
        let value = serde_json::to_vec(&index_data)?;
        self.db.put(&index_key, &value)?;
        Ok(())
    }

    fn entity_matches_filter(&self, entity: &Entity, filter: &EntityFilter) -> bool {
        if let Some(ref types) = filter.entity_types {
            if !types.contains(&entity.entity_type_name().to_string()) {
                return false;
            }
        }

        if let Some(ref tags) = filter.tags {
            if !tags.iter().any(|tag| entity.tags.contains(tag)) {
                return false;
            }
        }

        if let Some(ref name_filter) = filter.name_contains {
            if !entity.name().to_lowercase().contains(&name_filter.to_lowercase()) {
                return false;
            }
        }

        if let Some(has_relationships) = filter.has_relationships {
            let has_rels = !entity.relationships().is_empty();
            if has_relationships != has_rels {
                return false;
            }
        }

        true
    }
}