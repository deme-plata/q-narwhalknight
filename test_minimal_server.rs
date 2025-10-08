use axum::{
    response::Json,
    routing::get,
    Router,
};
use serde_json::{json, Value};

async fn test_handler() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn network_peers() -> Json<Value> {
    Json(json!({"peers": []}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/test", get(test_handler))
        .route("/api/v1/network/peers", get(network_peers))
        .route("/api/v1/network/peers", get(network_peers)); // Intentional duplicate

    println!("This should panic with route conflict...");
    
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}