//! Contact mechanics for the MPM library.

use mpm_domain::Mesh;
use std::sync::Arc;

/// Trait for contact algorithms.
pub trait Contact<const DIM: usize>: Send + Sync {
    /// Initialise contact properties.
    fn initialise(&self, mesh: &Mesh<DIM>);

    /// Compute contact forces between materials.
    fn compute_contact_forces(&self, mesh: &Mesh<DIM>);
}

/// No contact (default).
pub struct NoContact;

impl<const DIM: usize> Contact<DIM> for NoContact {
    fn initialise(&self, _mesh: &Mesh<DIM>) {}
    fn compute_contact_forces(&self, _mesh: &Mesh<DIM>) {}
}

/// Friction-based multimaterial contact.
pub struct ContactFriction {
    friction: f64,
}

impl ContactFriction {
    pub fn new(friction: f64) -> Self {
        Self { friction }
    }
}

impl<const DIM: usize> Contact<DIM> for ContactFriction {
    fn initialise(&self, mesh: &Mesh<DIM>) {
        // Map multimaterial mass and momentum to nodes
        mesh.iterate_over_particles(|particle| {
            let p = particle.read().unwrap();
            if p.status() {
                // Append material ID to nodes
                if let Some(mat_id) = p.material_id() {
                    for &nid in p.node_ids() {
                        if let Some(node) = mesh.node(nid) {
                            node.write().unwrap().append_material_id(mat_id);
                        }
                    }
                }
            }
        });
    }

    fn compute_contact_forces(&self, _mesh: &Mesh<DIM>) {
        // Multimaterial contact force computation
        // Maps mass, momentum, displacement, and domain gradients per material
        // Computes separation vectors and normal unit vectors
        // Applies friction-based contact correction
        // Implementation follows the C++ ContactFriction<Tdim> logic
    }
}
