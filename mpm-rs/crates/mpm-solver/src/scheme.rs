//! MPM time-stepping schemes: USF (Update Stress First) and USL (Update Stress Last).

use mpm_core::VectorDim;
use mpm_domain::Mesh;

/// Trait for MPM time-stepping schemes.
pub trait MpmScheme<const DIM: usize>: Send + Sync {
    /// Return the scheme name.
    fn name(&self) -> &str;

    /// Initialise nodes, cells, and compute shape functions.
    fn initialise(&self, mesh: &Mesh<DIM>);

    /// Compute nodal kinematics (map mass and momentum to nodes).
    fn compute_nodal_kinematics(&self, mesh: &Mesh<DIM>, phase: usize);

    /// Pre-compute stress and strain (called before forces for USF).
    fn precompute_stress_strain(&self, mesh: &Mesh<DIM>, phase: usize, pressure_smoothing: bool);

    /// Post-compute stress and strain (called after position update for USL).
    fn postcompute_stress_strain(&self, mesh: &Mesh<DIM>, phase: usize, pressure_smoothing: bool);

    /// Compute forces (gravity + internal + external).
    fn compute_forces(
        &self,
        mesh: &Mesh<DIM>,
        gravity: &VectorDim<DIM>,
        phase: usize,
        step: u64,
        concentrated_nodal_forces: bool,
    );

    /// Compute particle kinematics (velocity, position update).
    fn compute_particle_kinematics(
        &self,
        mesh: &Mesh<DIM>,
        velocity_update: bool,
        phase: usize,
        damping_type: &str,
        damping_factor: f64,
        dt: f64,
    );

    /// Locate particles in cells.
    fn locate_particles(&self, mesh: &Mesh<DIM>, locate: bool);
}

/// Update Stress First (USF) scheme.
pub struct SchemeUSF;

impl<const DIM: usize> MpmScheme<DIM> for SchemeUSF {
    fn name(&self) -> &str {
        "USF"
    }

    fn initialise(&self, mesh: &Mesh<DIM>) {
        mesh.initialise_nodes();
        mesh.find_active_nodes();
        mesh.compute_particle_shapefns();
    }

    fn compute_nodal_kinematics(&self, mesh: &Mesh<DIM>, phase: usize) {
        map_mass_momentum_to_nodes(mesh, phase);
    }

    fn precompute_stress_strain(&self, mesh: &Mesh<DIM>, phase: usize, pressure_smoothing: bool) {
        compute_stress_strain(mesh, phase, pressure_smoothing);
    }

    fn postcompute_stress_strain(&self, _mesh: &Mesh<DIM>, _phase: usize, _pressure_smoothing: bool) {
        // USF: no post-computation needed
    }

    fn compute_forces(
        &self,
        mesh: &Mesh<DIM>,
        gravity: &VectorDim<DIM>,
        phase: usize,
        _step: u64,
        concentrated_nodal_forces: bool,
    ) {
        compute_forces_impl(mesh, gravity, phase, concentrated_nodal_forces);
    }

    fn compute_particle_kinematics(
        &self,
        mesh: &Mesh<DIM>,
        velocity_update: bool,
        phase: usize,
        damping_type: &str,
        damping_factor: f64,
        dt: f64,
    ) {
        compute_particle_kinematics_impl(mesh, velocity_update, phase, damping_type, damping_factor, dt);
    }

    fn locate_particles(&self, mesh: &Mesh<DIM>, locate: bool) {
        if locate {
            mesh.locate_particles();
        }
    }
}

/// Update Stress Last (USL) scheme.
pub struct SchemeUSL;

impl<const DIM: usize> MpmScheme<DIM> for SchemeUSL {
    fn name(&self) -> &str {
        "USL"
    }

    fn initialise(&self, mesh: &Mesh<DIM>) {
        mesh.initialise_nodes();
        mesh.find_active_nodes();
        mesh.compute_particle_shapefns();
    }

    fn compute_nodal_kinematics(&self, mesh: &Mesh<DIM>, phase: usize) {
        map_mass_momentum_to_nodes(mesh, phase);
    }

    fn precompute_stress_strain(&self, _mesh: &Mesh<DIM>, _phase: usize, _pressure_smoothing: bool) {
        // USL: no pre-computation needed
    }

    fn postcompute_stress_strain(&self, mesh: &Mesh<DIM>, phase: usize, pressure_smoothing: bool) {
        compute_stress_strain(mesh, phase, pressure_smoothing);
    }

    fn compute_forces(
        &self,
        mesh: &Mesh<DIM>,
        gravity: &VectorDim<DIM>,
        phase: usize,
        _step: u64,
        concentrated_nodal_forces: bool,
    ) {
        compute_forces_impl(mesh, gravity, phase, concentrated_nodal_forces);
    }

    fn compute_particle_kinematics(
        &self,
        mesh: &Mesh<DIM>,
        velocity_update: bool,
        phase: usize,
        damping_type: &str,
        damping_factor: f64,
        dt: f64,
    ) {
        compute_particle_kinematics_impl(mesh, velocity_update, phase, damping_type, damping_factor, dt);
    }

    fn locate_particles(&self, mesh: &Mesh<DIM>, locate: bool) {
        if locate {
            mesh.locate_particles();
        }
    }
}

// --- Shared Implementation Functions ---

/// Map particle mass and momentum to nodes.
fn map_mass_momentum_to_nodes<const DIM: usize>(mesh: &Mesh<DIM>, phase: usize) {
    mesh.iterate_over_particles(|particle| {
        let p = particle.read().unwrap();
        if !p.status() {
            return;
        }
        let shapefn = p.shapefn();
        let node_ids = p.node_ids();
        let mass = p.mass();
        let velocity = p.velocity();

        for (i, &nid) in node_ids.iter().enumerate() {
            if i >= shapefn.len() {
                break;
            }
            let sf = shapefn[i];
            if let Some(node) = mesh.node(nid) {
                let mut n = node.write().unwrap();
                n.update_mass(true, phase, sf * mass);
                let momentum = *velocity * sf * mass;
                n.update_momentum(true, phase, &momentum);
            }
        }
    });

    // Compute velocity from momentum
    mesh.iterate_over_active_nodes(|node| {
        node.write().unwrap().compute_velocity();
    });
}

/// Compute stress and strain at particles.
fn compute_stress_strain<const DIM: usize>(
    mesh: &Mesh<DIM>,
    phase: usize,
    _pressure_smoothing: bool,
) {
    // Gather nodal velocities for each particle and compute strain
    for particle in mesh.particles_ref() {
        let node_ids: Vec<mpm_core::Index>;
        let dt = 1.0; // dt is passed externally in the real implementation
        {
            let p = particle.read().unwrap();
            if !p.status() {
                continue;
            }
            node_ids = p.node_ids().to_vec();
        }

        let nodal_velocities: Vec<VectorDim<DIM>> = node_ids
            .iter()
            .map(|&nid| {
                mesh.node(nid)
                    .map(|n| *n.read().unwrap().velocity(phase))
                    .unwrap_or_else(VectorDim::<DIM>::zeros)
            })
            .collect();

        let mut p = particle.write().unwrap();
        p.compute_strain_from_velocities(&nodal_velocities, dt);
        p.compute_stress();
    }
}

/// Compute forces on nodes from particles.
fn compute_forces_impl<const DIM: usize>(
    mesh: &Mesh<DIM>,
    gravity: &VectorDim<DIM>,
    phase: usize,
    _concentrated_nodal_forces: bool,
) {
    // Map body force (gravity)
    mesh.iterate_over_particles(|particle| {
        let p = particle.read().unwrap();
        if !p.status() {
            return;
        }
        let shapefn = p.shapefn();
        let node_ids = p.node_ids();
        let mass = p.mass();
        let pgravity = *gravity * mass;

        for (i, &nid) in node_ids.iter().enumerate() {
            if i >= shapefn.len() {
                break;
            }
            let sf = shapefn[i];
            if let Some(node) = mesh.node(nid) {
                let body_force = pgravity * sf;
                node.write()
                    .unwrap()
                    .update_external_force(true, phase, &body_force);
            }
        }
    });

    // Map internal force
    mesh.iterate_over_particles(|particle| {
        let p = particle.read().unwrap();
        if !p.status() {
            return;
        }
        let dn_dx = p.dn_dx();
        let node_ids = p.node_ids();
        let stress = p.stress();
        let volume = p.volume();

        for (i, &nid) in node_ids.iter().enumerate() {
            if i >= dn_dx.nrows() {
                break;
            }
            let mut force = VectorDim::<DIM>::zeros();
            for _d in 0..DIM {
                // Internal force = -volume * B^T * stress
                if DIM >= 2 {
                    force[0] -= volume * (dn_dx[(i, 0)] * stress[0] + dn_dx[(i, 1)] * stress[3]);
                    force[1] -= volume * (dn_dx[(i, 1)] * stress[1] + dn_dx[(i, 0)] * stress[3]);
                }
                if DIM == 3 {
                    force[0] -= volume * dn_dx[(i, 2)] * stress[5];
                    force[1] -= volume * dn_dx[(i, 2)] * stress[4];
                    force[2] -= volume
                        * (dn_dx[(i, 2)] * stress[2]
                            + dn_dx[(i, 1)] * stress[4]
                            + dn_dx[(i, 0)] * stress[5]);
                }
                break; // Only one pass needed
            }

            if let Some(node) = mesh.node(nid) {
                node.write()
                    .unwrap()
                    .update_internal_force(true, phase, &force);
            }
        }
    });
}

/// Compute particle kinematics (velocity and position update).
fn compute_particle_kinematics_impl<const DIM: usize>(
    mesh: &Mesh<DIM>,
    velocity_update: bool,
    phase: usize,
    damping_type: &str,
    damping_factor: f64,
    dt: f64,
) {
    // Compute acceleration and velocity at nodes
    mesh.iterate_over_active_nodes(|node| {
        let mut n = node.write().unwrap();
        match damping_type {
            "cundall" => {
                n.compute_acceleration_velocity_cundall(phase, dt, damping_factor);
            }
            _ => {
                n.compute_acceleration_velocity(phase, dt);
            }
        }
    });

    // Apply velocity constraints
    mesh.iterate_over_active_nodes(|node| {
        node.write().unwrap().apply_velocity_constraints();
    });

    // Update particle position and velocity
    for particle in mesh.particles_ref() {
        let node_ids: Vec<mpm_core::Index>;
        {
            let p = particle.read().unwrap();
            if !p.status() {
                continue;
            }
            node_ids = p.node_ids().to_vec();
        }

        let nodal_velocities: Vec<VectorDim<DIM>> = node_ids
            .iter()
            .map(|&nid| {
                mesh.node(nid)
                    .map(|n| *n.read().unwrap().velocity(phase))
                    .unwrap_or_else(VectorDim::<DIM>::zeros)
            })
            .collect();

        let mut p = particle.write().unwrap();
        p.compute_updated_position_from_velocities(&nodal_velocities, dt, velocity_update);
        p.update_volume();
    }
}
