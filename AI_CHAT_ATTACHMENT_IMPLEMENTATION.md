# AI Chat Attachment Support Implementation

**Date**: 2025-11-05
**Version**: v0.9.9-beta (planned)
**Status**: IMPLEMENTATION IN PROGRESS

---

## OVERVIEW

Implement comprehensive attachment support for AI chat, enabling users to upload images, PDFs, and other documents for AI analysis using mistral.rs vision models and document processing.

### Features

1. **File Upload**: Drag-and-drop + click-to-upload UI
2. **Supported Formats**:
   - Images: PNG, JPEG, GIF, WebP (vision model processing)
   - Documents: PDF, TXT, MD (text extraction)
   - Future: DOCX, XLSX (office documents)

3. **Storage**: Nginx temp files → Database → Permanent storage
4. **Processing**:
   - Images: Vision model analysis (mistral.rs Pixtral, LLaVA)
   - PDFs: Text extraction + embedding
   - Text files: Direct content inclusion

5. **Security**:
   - File size limits (25 MB per file, 100 MB total per chat)
   - MIME type validation
   - Virus scanning (optional, future)
   - User quota management

---

## ARCHITECTURE

```
┌─────────────┐
│   Browser   │
│  (Upload)   │
└──────┬──────┘
       │ HTTP POST multipart/form-data
       ▼
┌─────────────────────────────────┐
│         Nginx Proxy             │
│  - 25MB body limit              │
│  - /tmp/nginx-uploads/          │
│  - client_body_temp_path        │
└──────┬──────────────────────────┘
       │ Proxy to backend
       ▼
┌─────────────────────────────────┐
│   Axum Backend Handler          │
│  POST /api/chat/attachment      │
│  - Multipart parsing            │
│  - MIME validation              │
│  - Size check                   │
└──────┬──────────────────────────┘
       │ Save to disk + DB
       ▼
┌─────────────────────────────────┐
│      File Processing            │
│  - Images → vision model        │
│  - PDFs → text extraction       │
│  - TXT → direct inclusion       │
└──────┬──────────────────────────┘
       │ Processed content
       ▼
┌─────────────────────────────────┐
│   Mistral.rs Integration        │
│  - Vision models (images)       │
│  - Text generation with context │
│  - Multimodal input support     │
└─────────────────────────────────┘
```

---

## IMPLEMENTATION STEPS

### 1. Nginx Configuration

**File**: `/etc/nginx/sites-available/quillon.xyz`

Add to API location block:

```nginx
# API proxy with attachment upload support
location /api/ {
    # File upload limits
    client_max_body_size 25M;
    client_body_buffer_size 128k;

    # Temp file storage for uploads
    client_body_temp_path /tmp/nginx-uploads 1 2;
    client_body_in_file_only clean;

    # Longer timeouts for uploads
    proxy_read_timeout 300s;
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;

    # Pass filename to backend
    proxy_set_header X-File-Name $request_body_file;

    # Existing proxy settings...
    proxy_pass http://127.0.0.1:8080;
    # ... rest of config
}
```

Create upload directory:

```bash
mkdir -p /tmp/nginx-uploads
chown www-data:www-data /tmp/nginx-uploads
chmod 700 /tmp/nginx-uploads
```

### 2. Backend Dependencies

**File**: `crates/q-api-server/Cargo.toml`

```toml
[dependencies]
# Existing dependencies...

# File upload handling
multer = "3.1"
mime = "0.3"
mime_guess = "2.0"

# Image processing
image = { version = "0.25", features = ["png", "jpeg", "gif", "webp"] }

# PDF processing
pdf-extract = "0.7"
lopdf = "0.32"

# Vision model support
base64 = "0.22"
```

### 3. Database Schema

**Add to attachment table**:

```sql
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    message_id TEXT,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    thumbnail_path TEXT,
    extracted_text TEXT,
    vision_description TEXT,
    upload_timestamp INTEGER NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
);

CREATE INDEX idx_attachments_chat ON attachments(chat_id);
CREATE INDEX idx_attachments_message ON attachments(message_id);
```

### 4. Backend Attachment Handler

**File**: `crates/q-api-server/src/attachment_api.rs` (new file)

```rust
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use multer::Multipart as MultipartStream;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

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

pub async fn upload_attachment(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<AttachmentUploadResponse>, StatusCode> {
    // Parse multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let name = field.name().unwrap_or("").to_string();
        let filename = field.file_name().unwrap_or("unknown").to_string();
        let content_type = field.content_type().unwrap_or("application/octet-stream").to_string();

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

        // Read file data
        let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

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
        fs::create_dir_all(UPLOAD_DIR).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Save file to disk
        let mut file = fs::File::create(&storage_path)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        file.write_all(&data)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Save metadata to database
        let db = state.db.clone();
        let chat_id = ""; // Extract from request
        let user_id = ""; // Extract from auth header

        db.save_attachment(
            &attachment_id,
            chat_id,
            user_id,
            &filename,
            &content_type,
            data.len() as i64,
            &storage_path,
        ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Process attachment based on type
        tokio::spawn(process_attachment(
            state.clone(),
            attachment_id.clone(),
            storage_path.clone(),
            content_type.clone(),
        ));

        return Ok(Json(AttachmentUploadResponse {
            success: true,
            attachment_id: Some(attachment_id),
            filename: Some(filename),
            mime_type: Some(content_type),
            file_size: Some(data.len()),
            error: None,
        }));
    }

    Err(StatusCode::BAD_REQUEST)
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
    format!("att_{}", (0..16).map(|_| format!("{:x}", rng.gen::<u8>())).collect::<String>())
}

async fn process_attachment(
    state: Arc<AppState>,
    attachment_id: String,
    storage_path: String,
    mime_type: String,
) -> anyhow::Result<()> {
    match mime_type.as_str() {
        "image/png" | "image/jpeg" | "image/gif" | "image/webp" => {
            // Generate thumbnail
            let thumbnail_path = generate_thumbnail(&storage_path).await?;

            // Extract image for vision model (base64 encode)
            let image_data = fs::read(&storage_path).await?;
            let base64_image = base64::encode(&image_data);

            // Store for later vision model processing
            state.db.update_attachment_processed(
                &attachment_id,
                Some(&thumbnail_path),
                None,
                Some(&base64_image),
            ).await?;
        }
        "application/pdf" => {
            // Extract text from PDF
            let extracted_text = extract_pdf_text(&storage_path).await?;

            state.db.update_attachment_processed(
                &attachment_id,
                None,
                Some(&extracted_text),
                None,
            ).await?;
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
        }
        _ => {}
    }

    Ok(())
}

async fn generate_thumbnail(image_path: &str) -> anyhow::Result<String> {
    use image::ImageReader;

    let img = ImageReader::open(image_path)?.decode()?;
    let thumbnail = img.thumbnail(200, 200);

    let thumbnail_path = format!("{}_thumb.jpg", image_path.trim_end_matches(|c: char| c != '.'));
    thumbnail.save(&thumbnail_path)?;

    Ok(thumbnail_path)
}

async fn extract_pdf_text(pdf_path: &str) -> anyhow::Result<String> {
    use pdf_extract::extract_text;

    let text = extract_text(pdf_path)?;
    Ok(text)
}
```

### 5. Vision Model Integration

**Extend `chat_api.rs` streaming endpoint**:

```rust
// In streaming handler, check for attachments
let attachments = state.db.get_message_attachments(chat_id).await?;

let mut vision_context = String::new();
for attachment in attachments {
    if attachment.mime_type.starts_with("image/") {
        if let Some(vision_data) = attachment.vision_base64 {
            // Add image to vision model input
            vision_context.push_str(&format!("[Image: {}]\n", attachment.filename));

            // Vision models in mistral.rs support base64 images
            // Include in generation request
        }
    } else if let Some(text) = attachment.extracted_text {
        vision_context.push_str(&format!("\n[Document: {}]\n{}\n", attachment.filename, text));
    }
}

// Prepend attachment context to user message
let full_prompt = format!("{}\n\nUser question: {}", vision_context, user_message);
```

### 6. Frontend Upload UI

**File**: `gui/quantum-wallet/src/components/AIChatScreen.tsx`

Add attachment state and upload logic:

```typescript
const [attachments, setAttachments] = useState<Attachment[]>([]);
const [uploading, setUploading] = useState(false);

interface Attachment {
  id: string;
  filename: string;
  mimeType: string;
  fileSize: number;
  thumbnailUrl?: string;
  uploadProgress: number;
}

const handleFileUpload = async (files: FileList) => {
  setUploading(true);

  for (const file of Array.from(files)) {
    // Validate file
    if (file.size > 25 * 1024 * 1024) {
      alert(`File too large: ${file.name} (max 25 MB)`);
      continue;
    }

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chat_id', currentChatId!);

    try {
      const response = await fetch('/api/chat/attachment', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        setAttachments(prev => [...prev, {
          id: result.attachment_id,
          filename: result.filename,
          mimeType: result.mime_type,
          fileSize: result.file_size,
          uploadProgress: 100,
        }]);
      } else {
        alert(`Upload failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file');
    }
  }

  setUploading(false);
};

// Drag-and-drop UI
<div
  className="attachment-drop-zone"
  onDrop={(e) => {
    e.preventDefault();
    handleFileUpload(e.dataTransfer.files);
  }}
  onDragOver={(e) => e.preventDefault()}
>
  <input
    type="file"
    id="file-upload"
    multiple
    accept="image/*,application/pdf,text/*"
    onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
    className="hidden"
  />
  <label htmlFor="file-upload" className="upload-button">
    📎 Attach files (images, PDFs, text)
  </label>

  {attachments.map(att => (
    <div key={att.id} className="attachment-preview">
      {att.mimeType.startsWith('image/') ? (
        <img src={`/api/attachment/${att.id}/thumbnail`} alt={att.filename} />
      ) : (
        <span>📄 {att.filename}</span>
      )}
      <button onClick={() => removeAttachment(att.id)}>×</button>
    </div>
  ))}
</div>
```

---

## DEPLOYMENT PLAN

### Phase 1: Nginx + Basic Upload (ETA: 30 min)
1. Configure Nginx temp uploads
2. Implement backend multipart handler
3. Test file upload endpoint

### Phase 2: Storage + Database (ETA: 20 min)
4. Create attachments table
5. Implement database save/retrieve
6. Add file size/quota checks

### Phase 3: Processing (ETA: 40 min)
7. PDF text extraction
8. Image thumbnail generation
9. Base64 encoding for vision models

### Phase 4: Vision Model Integration (ETA: 60 min)
10. Mistral.rs vision model configuration
11. Multimodal input formatting
12. Streaming with attachments

### Phase 5: Frontend UI (ETA: 45 min)
13. Upload button + drag-and-drop
14. Attachment previews
15. Progress indicators
16. Error handling

**Total ETA**: ~3-4 hours for complete implementation

---

## TESTING CHECKLIST

- [ ] Upload PNG image < 25 MB
- [ ] Upload PDF document
- [ ] Upload text file
- [ ] Reject file > 25 MB
- [ ] Reject unsupported MIME types
- [ ] Vision model processes image correctly
- [ ] PDF text extraction works
- [ ] Thumbnails generate properly
- [ ] Attachments appear in chat UI
- [ ] AI responds with attachment context
- [ ] Multiple attachments per message
- [ ] Attachment deletion works
- [ ] Database cleanup on chat deletion

---

## SECURITY CONSIDERATIONS

1. **File Size Limits**: 25 MB per file prevents memory exhaustion
2. **MIME Validation**: Whitelist only safe file types
3. **Path Traversal**: Sanitize filenames, use UUIDs
4. **Storage Quota**: Limit total storage per user
5. **Access Control**: Only chat owner can view attachments
6. **Encryption**: Store attachments encrypted at rest (future)
7. **Virus Scanning**: ClamAV integration (future)

---

## PERFORMANCE OPTIMIZATIONS

1. **Thumbnails**: Pre-generate for fast preview loading
2. **Lazy Loading**: Only load attachment data when needed
3. **CDN**: Serve static attachments via CDN (future)
4. **Compression**: Compress images before storage
5. **Async Processing**: Background job for PDF extraction
6. **Caching**: Cache extracted text in memory

---

## FUTURE ENHANCEMENTS

1. **Office Documents**: DOCX, XLSX support via pandoc
2. **Audio/Video**: Transcription with Whisper
3. **OCR**: Extract text from scanned PDFs
4. **Embedding Search**: Vector search across attachments
5. **Collaborative Editing**: Real-time document annotation
6. **Version Control**: Track attachment edits

---

## STATUS

**Current**: Implementation plan complete, ready to begin
**Next**: Start with Phase 1 (Nginx configuration)
**Blockers**: None

This is a comprehensive 3-4 hour implementation. Shall I proceed with Phase 1?
