//! Explicit MPM solver.

use crate::scheme::{MpmScheme, SchemeUSF, SchemeUSL};
use mpm_core::VectorDim;
use mpm_domain::Mesh;
use mpm_io::config::MpmConfig;

/// Explicit MPM solver with configurable time-stepping scheme.
pub struct MpmExplicit<const DIM: usize> {
    mesh: Mesh<DIM>,
    scheme: Box<dyn MpmScheme<DIM>>,
    dt: f64,
    nsteps: u64,
    step: u64,
    output_steps: u64,
    velocity_update: bool,
    gravity: VectorDim<DIM>,
    damping_type: String,
    damping_factor: f64,
    pressure_smoothing: bool,
    locate_particles: bool,
    concentrated_nodal_forces: bool,
    uuid: String,
}

impl<const DIM: usize> MpmExplicit<DIM> {
    /// Create a new explicit MPM solver from configuration.
    pub fn new(config: &MpmConfig, mesh: Mesh<DIM>) -> Self {
        let scheme_name = config.analysis.mpm_scheme.to_lowercase();
        let scheme: Box<dyn MpmScheme<DIM>> = match scheme_name.as_str() {
            "usf" => Box::new(SchemeUSF),
            "usl" | _ => Box::new(SchemeUSL),
        };

        let mut gravity = VectorDim::<DIM>::zeros();
        for (d, &g) in config.analysis.gravity.iter().enumerate() {
            if d < DIM {
                gravity[d] = g;
            }
        }

        let (damping_type, damping_factor) = match &config.analysis.damping {
            Some(d) => (d.damping_type.clone(), d.damping_factor),
            None => ("none".to_string(), 0.0),
        };

        let output_steps = config
            .post_processing
            .as_ref()
            .map(|pp| pp.output_steps)
            .unwrap_or(0);

        Self {
            mesh,
            scheme,
            dt: config.analysis.dt,
            nsteps: config.analysis.nsteps,
            step: 0,
            output_steps,
            velocity_update: config.analysis.velocity_update,
            gravity,
            damping_type,
            damping_factor,
            pressure_smoothing: config.analysis.pressure_smoothing.unwrap_or(false),
            locate_particles: config.analysis.locate_particles.unwrap_or(true),
            concentrated_nodal_forces: false,
            uuid: config
                .analysis
                .uuid
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        }
    }

    /// Run the explicit MPM solver.
    pub fn solve(&mut self) -> Result<(), mpm_core::MpmError> {
        tracing::info!(
            "MPM Explicit Analysis: {} steps, dt = {}, scheme = {}",
            self.nsteps,
            self.dt,
            self.scheme.name()
        );

        let phase = 0;

        for step in 0..self.nsteps {
            self.step = step;

            // 1. Initialise (reset nodes, compute shape functions)
            self.scheme.initialise(&self.mesh);

            // 2. Map mass and momentum to nodes
            self.scheme.compute_nodal_kinematics(&self.mesh, phase);

            // 3. Pre-compute stress/strain (USF)
            self.scheme.precompute_stress_strain(
                &self.mesh,
                phase,
                self.pressure_smoothing,
            );

            // 4. Compute forces
            self.scheme.compute_forces(
                &self.mesh,
                &self.gravity,
                phase,
                step,
                self.concentrated_nodal_forces,
            );

            // 5. Compute particle kinematics
            self.scheme.compute_particle_kinematics(
                &self.mesh,
                self.velocity_update,
                phase,
                &self.damping_type,
                self.damping_factor,
                self.dt,
            );

            // 6. Post-compute stress/strain (USL)
            self.scheme.postcompute_stress_strain(
                &self.mesh,
                phase,
                self.pressure_smoothing,
            );

            // 7. Locate particles
            self.scheme.locate_particles(&self.mesh, self.locate_particles);

            // 8. Output
            if self.output_steps > 0 && step % self.output_steps == 0 {
                tracing::info!("Step {}/{}", step, self.nsteps);
            }
        }

        tracing::info!("MPM Explicit Analysis complete");
        Ok(())
    }

    /// Get a reference to the mesh.
    pub fn mesh(&self) -> &Mesh<DIM> {
        &self.mesh
    }

    /// Get a mutable reference to the mesh.
    pub fn mesh_mut(&mut self) -> &mut Mesh<DIM> {
        &mut self.mesh
    }

    pub fn uuid(&self) -> &str {
        &self.uuid
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }
}
