//! MPM entry point.
//!
//! Parses command-line arguments, reads configuration, creates the solver,
//! and runs the analysis.

use clap::Parser;
use mpm_io::config::{Cli, MpmConfig};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialise logging
    tracing_subscriber::fmt::init();

    // Parse command-line arguments
    let cli = Cli::parse();

    tracing::info!("CB-Geo MPM (Rust)");
    tracing::info!("Working directory: {:?}", cli.working_dir);

    // Set thread count
    if let Some(nthreads) = cli.parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build_global()
            .ok();
        tracing::info!("Using {} threads", nthreads);
    }

    // Read configuration
    let config_path = cli.working_dir.join(&cli.input_file);
    let config_str = fs::read_to_string(&config_path).map_err(|e| {
        format!(
            "Failed to read config file '{}': {}",
            config_path.display(),
            e
        )
    })?;

    let config: MpmConfig = serde_json::from_str(&config_str).map_err(|e| {
        format!("Failed to parse config: {}", e)
    })?;

    tracing::info!("Title: {}", config.title);
    tracing::info!("Analysis type: {}", config.analysis.analysis_type);

    // Determine dimension and run
    match config.analysis.analysis_type.as_str() {
        "MPMExplicit2D" => {
            run_2d(&cli, &config)?;
        }
        "MPMExplicit3D" => {
            run_3d(&cli, &config)?;
        }
        _ => {
            return Err(format!(
                "Unknown analysis type: {}",
                config.analysis.analysis_type
            )
            .into());
        }
    }

    Ok(())
}

fn run_2d(
    _cli: &Cli,
    config: &MpmConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use mpm_domain::Mesh;
    use mpm_solver::MpmExplicit;

    let mesh = Mesh::<2>::new(0, true);

    // TODO: initialise mesh from config (read nodes, cells, particles)
    // This would use mpm_io::mesh_ascii functions to read the mesh files
    // and populate the Mesh object

    let mut solver = MpmExplicit::new(config, mesh);

    tracing::info!("Starting 2D explicit MPM solver");
    solver.solve()?;

    Ok(())
}

fn run_3d(
    _cli: &Cli,
    config: &MpmConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use mpm_domain::Mesh;
    use mpm_solver::MpmExplicit;

    let mesh = Mesh::<3>::new(0, true);

    // TODO: initialise mesh from config
    let mut solver = MpmExplicit::new(config, mesh);

    tracing::info!("Starting 3D explicit MPM solver");
    solver.solve()?;

    Ok(())
}
