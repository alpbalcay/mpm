//! Configuration types for MPM analysis.

use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;

/// MPM command-line interface.
#[derive(Parser, Debug)]
#[command(name = "mpm", about = "CB-Geo Material Point Method")]
pub struct Cli {
    /// Working directory containing input files
    #[arg(short = 'f', long)]
    pub working_dir: PathBuf,

    /// Input JSON file name
    #[arg(short = 'i', long, default_value = "mpm.json")]
    pub input_file: String,

    /// Number of parallel threads
    #[arg(short = 'p', long)]
    pub parallel: Option<usize>,
}

/// Top-level MPM configuration from JSON.
#[derive(Debug, Deserialize)]
pub struct MpmConfig {
    pub title: String,
    pub mesh: MeshConfig,
    #[serde(default)]
    pub particles: Vec<ParticleGeneratorConfig>,
    pub materials: Vec<MaterialConfig>,
    pub analysis: AnalysisConfig,
    #[serde(default)]
    pub post_processing: Option<PostProcessingConfig>,
    #[serde(default)]
    pub math_functions: Vec<FunctionConfig>,
}

/// Mesh configuration.
#[derive(Debug, Deserialize)]
pub struct MeshConfig {
    pub mesh: String,
    #[serde(default = "default_isoparametric")]
    pub isoparametric: bool,
    #[serde(default = "default_check_duplicates")]
    pub check_duplicates: bool,
    #[serde(default)]
    pub cell_type: String,
    #[serde(default)]
    pub node_sets: Vec<EntitySetConfig>,
    #[serde(default)]
    pub cell_sets: Vec<EntitySetConfig>,
    #[serde(default)]
    pub boundary_conditions: Option<BoundaryConditionsConfig>,
}

fn default_isoparametric() -> bool { true }
fn default_check_duplicates() -> bool { true }

/// Entity set configuration (node/cell/particle sets).
#[derive(Debug, Deserialize)]
pub struct EntitySetConfig {
    pub id: u32,
    pub set: String,
}

/// Boundary conditions configuration.
#[derive(Debug, Deserialize)]
pub struct BoundaryConditionsConfig {
    #[serde(default)]
    pub velocity_constraints: Vec<ConstraintConfig>,
    #[serde(default)]
    pub friction_constraints: Vec<FrictionConfig>,
    #[serde(default)]
    pub nodal_euler_angles: Option<String>,
}

/// Velocity constraint configuration.
#[derive(Debug, Deserialize)]
pub struct ConstraintConfig {
    pub file: Option<String>,
    pub nset_id: Option<i32>,
    pub dir: Option<usize>,
    pub velocity: Option<f64>,
}

/// Friction constraint configuration.
#[derive(Debug, Deserialize)]
pub struct FrictionConfig {
    pub file: Option<String>,
    pub nset_id: Option<i32>,
    pub dir: Option<usize>,
    pub sign_n: Option<i32>,
    pub friction: Option<f64>,
}

/// Particle generator configuration.
#[derive(Debug, Deserialize)]
pub struct ParticleGeneratorConfig {
    #[serde(rename = "type")]
    pub generator_type: Option<String>,
    pub location: Option<String>,
    #[serde(default)]
    pub material_id: Vec<u32>,
    #[serde(default)]
    pub particle_sets: Vec<EntitySetConfig>,
    #[serde(default)]
    pub particles_volumes: Option<String>,
    #[serde(default)]
    pub particles_stresses: Option<String>,
    #[serde(default)]
    pub particles_cells: Option<String>,
    #[serde(default)]
    pub particle_type: Option<String>,
    pub io_type: Option<String>,
    pub pset_id: Option<u32>,
    pub cset_id: Option<i32>,
    pub nparticles_per_dir: Option<usize>,
}

/// Material configuration.
#[derive(Debug, Deserialize)]
pub struct MaterialConfig {
    pub id: u32,
    #[serde(rename = "type")]
    pub material_type: String,
    #[serde(flatten)]
    pub properties: serde_json::Value,
}

/// Analysis configuration.
#[derive(Debug, Deserialize)]
pub struct AnalysisConfig {
    #[serde(rename = "type")]
    pub analysis_type: String,
    #[serde(default = "default_scheme")]
    pub mpm_scheme: String,
    pub dt: f64,
    pub nsteps: u64,
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(default = "default_false")]
    pub velocity_update: bool,
    #[serde(default)]
    pub gravity: Vec<f64>,
    #[serde(default)]
    pub locate_particles: Option<bool>,
    #[serde(default)]
    pub pressure_smoothing: Option<bool>,
    #[serde(default)]
    pub damping: Option<DampingConfig>,
    #[serde(default)]
    pub resume: Option<ResumeConfig>,
}

fn default_scheme() -> String { "usf".to_string() }
fn default_false() -> bool { false }

/// Damping configuration.
#[derive(Debug, Deserialize)]
pub struct DampingConfig {
    #[serde(rename = "type")]
    pub damping_type: String,
    pub damping_factor: f64,
}

/// Checkpoint resume configuration.
#[derive(Debug, Deserialize)]
pub struct ResumeConfig {
    pub resume: bool,
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(default)]
    pub step: Option<u64>,
}

/// Post-processing configuration.
#[derive(Debug, Deserialize)]
pub struct PostProcessingConfig {
    #[serde(default)]
    pub output_steps: u64,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub vtk: Vec<String>,
    #[serde(default)]
    pub vtk_statevars: Vec<VtkStatevarConfig>,
}

/// VTK state variable output configuration.
#[derive(Debug, Deserialize)]
pub struct VtkStatevarConfig {
    pub phase_id: u32,
    pub statevars: Vec<String>,
}

/// Function configuration.
#[derive(Debug, Deserialize)]
pub struct FunctionConfig {
    pub id: u64,
    #[serde(rename = "type")]
    pub function_type: String,
    #[serde(flatten)]
    pub properties: serde_json::Value,
}
