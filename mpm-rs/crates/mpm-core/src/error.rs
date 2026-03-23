//! Error types for the MPM library.

use crate::Index;

#[derive(Debug, thiserror::Error)]
pub enum MpmError {
    #[error("Invalid material properties: {0}")]
    InvalidMaterial(String),

    #[error("Particle {0} not in any cell")]
    ParticleOutOfDomain(Index),

    #[error("Cell {0} not found")]
    CellNotFound(Index),

    #[error("Node {0} not found")]
    NodeNotFound(Index),

    #[error("Invalid element configuration: {0}")]
    InvalidElement(String),

    #[error("Invalid mesh: {0}")]
    InvalidMesh(String),

    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(String),

    #[error("Duplicate ID: {0}")]
    DuplicateId(Index),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),
}
