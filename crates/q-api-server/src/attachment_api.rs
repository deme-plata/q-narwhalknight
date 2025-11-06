// AI Chat Attachment Support - File Upload Handler
// Supports images, PDFs, and text files for vision model processing

use axum::{
    body::Bytes,
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use serde::{Deserialize, Serialize};

const MAX_FILE_SIZE: usize = 25 * 1024 * 1024; // 25 MB
const UPLOAD_DIR: &str = "/opt/orobit/shared/q-narwhalknight/data/attachments";

#[derive(Debug, Serialize, Deserialize)]
pub struct AttachmentUploadResponse {
    pub success: bool,
    pub attachment_id: Option<String>,
    pub filename: Option<String>,
    pub mime_type: Option<String>,
    pub file_size: Option<usize>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AttachmentMetadata {
    pub id: String,
    pub chat_id: String,
    pub user_id: String,
    pub filename: String,
    pub mime_type: String,
    pub file_size: i64,
    pub storage_path: String,
    pub thumbnail_path: Option<String>,
    pub extracted_text: Option<String>,
    pub vision_base64: Option<String>,
    pub upload_timestamp: i64,
    pub processed: bool,
}

pub async fn upload_attachment(
    State(state): State<Arc<crate::AppState>>,
    mut multipart: Multipart,
) -> Result<Json<AttachmentUploadResponse>, StatusCode> {
    // Parse multipart form data
    let mut chat_id: Option<String> = None;
    let mut file_data: Option<(String, String, Bytes)> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::error!("Failed to parse multipart field: {}", e);
        StatusCode::BAD_REQUEST
    })? {
        let name = field.name().map(|s| s.to_string()).unwrap_or_default();

        match name.as_str() {
            "chat_id" => {
                let value = field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?;
                chat_id = Some(value);
            }
            "file" => {
                let filename = field.file_name().unwrap_or("unknown").to_string();
                let content_type = field.content_type().unwrap_or("application/octet-stream").to_string();
                let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

                file_data = Some((filename, content_type, data));
            }
            _ => {}
        }
    }

    // Validate required fields
    let chat_id = chat_id.ok_or(StatusCode::BAD_REQUEST)?;
    let (filename, content_type, data) = file_data.ok_or(StatusCode::BAD_REQUEST)?;

    // Validate MIME type
    if !is_allowed_mime_type(&content_type) {
        return Ok(Json(AttachmentUploadResponse {
            success: false,
            attachment_id: None,
            filename: None,
            mime_type: None,
            file_size: None,
            error: Some(format!("File type not allowed: {}", content_type)),
        }));
    }

    // Check file size
    if data.len() > MAX_FILE_SIZE {
        return Ok(Json(AttachmentUploadResponse {
            success: false,
            attachment_id: None,
            filename: None,
            mime_type: None,
            file_size: None,
            error: Some(format!("File too large: {} bytes (max: {} MB)", data.len(), MAX_FILE_SIZE / 1024 / 1024)),
        }));
    }

    // Generate unique attachment ID
    let attachment_id = generate_attachment_id();
    let file_extension = PathBuf::from(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");

    let storage_path = format!("{}/{}.{}", UPLOAD_DIR, attachment_id, file_extension);

    // Ensure upload directory exists
    fs::create_dir_all(UPLOAD_DIR).await.map_err(|e| {
        tracing::error!("Failed to create upload directory: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Save file to disk
    let mut file = fs::File::create(&storage_path)
        .await
        .map_err(|e| {
            tracing::error!("Failed to create file: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    file.write_all(&data)
        .await
        .map_err(|e| {
            tracing::error!("Failed to write file: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Get user ID from wallet address (stored in localStorage on frontend)
    // For now, use a default user ID - will be enhanced with proper auth
    let user_id = "default";

    // Save metadata to database
    let db = state.db.clone();
    db.save_attachment(
        &attachment_id,
        &chat_id,
        user_id,
        &filename,
        &content_type,
        data.len() as i64,
        &storage_path,
    ).await.map_err(|e| {
        tracing::error!("Failed to save attachment metadata: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("📎 Attachment uploaded: {} ({} bytes, {})", filename, data.len(), content_type);

    // Process attachment based on type (async in background)
    let state_clone = state.clone();
    let attachment_id_clone = attachment_id.clone();
    let storage_path_clone = storage_path.clone();
    let content_type_clone = content_type.clone();

    tokio::spawn(async move {
        if let Err(e) = process_attachment(
            state_clone,
            attachment_id_clone,
            storage_path_clone,
            content_type_clone,
        ).await {
            tracing::error!("Failed to process attachment: {}", e);
        }
    });

    Ok(Json(AttachmentUploadResponse {
        success: true,
        attachment_id: Some(attachment_id),
        filename: Some(filename),
        mime_type: Some(content_type),
        file_size: Some(data.len()),
        error: None,
    }))
}

fn is_allowed_mime_type(mime: &str) -> bool {
    matches!(
        mime,
        "image/png" | "image/jpeg" | "image/gif" | "image/webp" |
        "application/pdf" |
        "text/plain" | "text/markdown" | "text/csv"
    )
}

fn generate_attachment_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..16).map(|_| rng.gen()).collect();
    format!("att_{}", hex::encode(random_bytes))
}

async fn process_attachment(
    state: Arc<crate::AppState>,
    attachment_id: String,
    storage_path: String,
    mime_type: String,
) -> anyhow::Result<()> {
    tracing::info!("🔄 Processing attachment: {} ({})", attachment_id, mime_type);

    match mime_type.as_str() {
        "image/png" | "image/jpeg" | "image/gif" | "image/webp" => {
            // Generate thumbnail
            let thumbnail_path = match generate_thumbnail(&storage_path).await {
                Ok(path) => Some(path),
                Err(e) => {
                    tracing::warn!("Failed to generate thumbnail: {}", e);
                    None
                }
            };

            // Read image for vision model (base64 encode)
            let image_data = fs::read(&storage_path).await?;
            let base64_image = base64::encode(&image_data);

            // Store for later vision model processing
            state.db.update_attachment_processed(
                &attachment_id,
                thumbnail_path.as_deref(),
                None,
                Some(&base64_image),
            ).await?;

            tracing::info!("✅ Image processed: {} (thumbnail: {})", attachment_id, thumbnail_path.is_some());
        }
        "application/pdf" => {
            // Extract text from PDF
            let extracted_text = match extract_pdf_text(&storage_path).await {
                Ok(text) => text,
                Err(e) => {
                    tracing::warn!("Failed to extract PDF text: {}", e);
                    String::new()
                }
            };

            state.db.update_attachment_processed(
                &attachment_id,
                None,
                Some(&extracted_text),
                None,
            ).await?;

            tracing::info!("✅ PDF processed: {} ({} chars extracted)", attachment_id, extracted_text.len());
        }
        "text/plain" | "text/markdown" | "text/csv" => {
            // Read text content directly
            let content = fs::read_to_string(&storage_path).await?;

            state.db.update_attachment_processed(
                &attachment_id,
                None,
                Some(&content),
                None,
            ).await?;

            tracing::info!("✅ Text file processed: {} ({} chars)", attachment_id, content.len());
        }
        _ => {
            tracing::warn!("Unsupported MIME type for processing: {}", mime_type);
        }
    }

    Ok(())
}

async fn generate_thumbnail(image_path: &str) -> anyhow::Result<String> {
    use image::ImageReader;

    let img = ImageReader::open(image_path)?.decode()?;
    let thumbnail = img.thumbnail(200, 200);

    let path = PathBuf::from(image_path);
    let thumbnail_path = format!(
        "{}/{}_thumb.jpg",
        path.parent().unwrap().display(),
        path.file_stem().unwrap().to_str().unwrap()
    );

    thumbnail.save(&thumbnail_path)?;

    Ok(thumbnail_path)
}

async fn extract_pdf_text(pdf_path: &str) -> anyhow::Result<String> {
    use pdf_extract::extract_text;

    let text = extract_text(pdf_path)?;
    Ok(text)
}
