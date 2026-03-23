//! Core types and utilities for the MPM library.
//!
//! Provides type aliases for linear algebra types (via nalgebra),
//! error types, and common constants used throughout the codebase.

pub mod error;
pub mod types;

pub use error::MpmError;
pub use types::*;
