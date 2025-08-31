use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use q_api_server::{handlers, streaming, AppState, Config};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "q_api_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::from_env()?;
    info!("Starting Q-NarwhalKnight API Server on port {}", config.port);

    // Initialize application state
    let state = AppState::new(config.clone()).await?;
    let app_state = Arc::new(state);

    // Build the application router
    let app = Router::new()
        // Wallet endpoints
        .route("/api/v1/wallets", post(handlers::create_wallet))
        .route("/api/v1/wallets", get(handlers::list_wallets))
        .route("/api/v1/wallets/:id", get(handlers::get_wallet))
        .route("/api/v1/wallets/:id/sign", post(handlers::sign_transaction))
        
        // Chain endpoints
        .route("/api/v1/status", get(handlers::node_status))
        .route("/api/v1/transactions", post(handlers::submit_transaction))
        .route("/api/v1/transactions/:hash", get(handlers::get_transaction))
        .route("/api/v1/blocks/:height", get(handlers::get_block))
        
        // Real-time streaming endpoints
        .route("/api/v1/events", get(streaming::sse_events))
        .route("/api/v1/ws", get(streaming::websocket_handler))
        
        // Health and metrics
        .route("/health", get(handlers::health_check))
        .route("/metrics", get(handlers::metrics))
        
        // Add middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
        .with_state(app_state);

    // Start the server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", config.port)).await?;
    info!("API server listening on {}", listener.local_addr()?);
    
    axum::serve(listener, app).await?;

    Ok(())
}