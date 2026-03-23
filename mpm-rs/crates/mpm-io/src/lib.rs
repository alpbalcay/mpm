//! I/O module for MPM: configuration parsing, mesh reading, output writing.

pub mod config;
pub mod mesh_ascii;
pub mod vtk_writer;
pub mod hdf5_particle;

pub use config::{Cli, MpmConfig};
