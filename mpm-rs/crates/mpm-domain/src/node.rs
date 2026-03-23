//! Grid node for the MPM method.

use mpm_core::{Index, VectorDim, NPHASES};
use mpm_functions::Function;
use nalgebra::SMatrix;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A grid node that stores mass, momentum, and force data.
pub struct Node<const DIM: usize> {
    id: Index,
    coordinates: VectorDim<DIM>,
    status: bool,
    mass: [f64; NPHASES],
    volume: [f64; NPHASES],
    external_force: [VectorDim<DIM>; NPHASES],
    internal_force: [VectorDim<DIM>; NPHASES],
    pressure: [f64; NPHASES],
    velocity: [VectorDim<DIM>; NPHASES],
    momentum: [VectorDim<DIM>; NPHASES],
    acceleration: [VectorDim<DIM>; NPHASES],
    velocity_constraints: BTreeMap<usize, f64>,
    friction_constraint: Option<(usize, i32, f64)>,
    rotation_matrix: SMatrix<f64, DIM, DIM>,
    generic_boundary_constraints: bool,
    material_ids: BTreeSet<u32>,
    concentrated_force: [VectorDim<DIM>; NPHASES],
    force_function: Option<Arc<dyn Function>>,
    #[cfg(feature = "mpi")]
    mpi_ranks: BTreeSet<u32>,
    ghost_id: Index,
}

impl<const DIM: usize> Node<DIM> {
    pub fn new(id: Index, coordinates: VectorDim<DIM>) -> Self {
        Self {
            id,
            coordinates,
            status: false,
            mass: [0.0; NPHASES],
            volume: [0.0; NPHASES],
            external_force: [VectorDim::<DIM>::zeros(); NPHASES],
            internal_force: [VectorDim::<DIM>::zeros(); NPHASES],
            pressure: [0.0; NPHASES],
            velocity: [VectorDim::<DIM>::zeros(); NPHASES],
            momentum: [VectorDim::<DIM>::zeros(); NPHASES],
            acceleration: [VectorDim::<DIM>::zeros(); NPHASES],
            velocity_constraints: BTreeMap::new(),
            friction_constraint: None,
            rotation_matrix: SMatrix::<f64, DIM, DIM>::identity(),
            generic_boundary_constraints: false,
            material_ids: BTreeSet::new(),
            concentrated_force: [VectorDim::<DIM>::zeros(); NPHASES],
            force_function: None,
            #[cfg(feature = "mpi")]
            mpi_ranks: BTreeSet::new(),
            ghost_id: std::u64::MAX,
        }
    }

    pub fn id(&self) -> Index {
        self.id
    }

    pub fn coordinates(&self) -> &VectorDim<DIM> {
        &self.coordinates
    }

    pub fn assign_coordinates(&mut self, coord: VectorDim<DIM>) {
        self.coordinates = coord;
    }

    pub fn status(&self) -> bool {
        self.status
    }

    pub fn assign_status(&mut self, status: bool) {
        self.status = status;
    }

    /// Reset node state for a new time step.
    pub fn initialise(&mut self) {
        self.mass = [0.0; NPHASES];
        self.volume = [0.0; NPHASES];
        self.external_force = [VectorDim::<DIM>::zeros(); NPHASES];
        self.internal_force = [VectorDim::<DIM>::zeros(); NPHASES];
        self.pressure = [0.0; NPHASES];
        self.velocity = [VectorDim::<DIM>::zeros(); NPHASES];
        self.momentum = [VectorDim::<DIM>::zeros(); NPHASES];
        self.acceleration = [VectorDim::<DIM>::zeros(); NPHASES];
        self.status = false;
    }

    pub fn update_mass(&mut self, update: bool, phase: usize, mass: f64) {
        if update {
            self.mass[phase] += mass;
        } else {
            self.mass[phase] = mass;
        }
    }

    pub fn mass(&self, phase: usize) -> f64 {
        self.mass[phase]
    }

    pub fn update_volume(&mut self, update: bool, phase: usize, volume: f64) {
        if update {
            self.volume[phase] += volume;
        } else {
            self.volume[phase] = volume;
        }
    }

    pub fn volume(&self, phase: usize) -> f64 {
        self.volume[phase]
    }

    pub fn update_external_force(&mut self, update: bool, phase: usize, force: &VectorDim<DIM>) {
        if update {
            self.external_force[phase] += force;
        } else {
            self.external_force[phase] = *force;
        }
    }

    pub fn external_force(&self, phase: usize) -> &VectorDim<DIM> {
        &self.external_force[phase]
    }

    pub fn update_internal_force(&mut self, update: bool, phase: usize, force: &VectorDim<DIM>) {
        if update {
            self.internal_force[phase] += force;
        } else {
            self.internal_force[phase] = *force;
        }
    }

    pub fn internal_force(&self, phase: usize) -> &VectorDim<DIM> {
        &self.internal_force[phase]
    }

    pub fn update_momentum(&mut self, update: bool, phase: usize, momentum: &VectorDim<DIM>) {
        if update {
            self.momentum[phase] += momentum;
        } else {
            self.momentum[phase] = *momentum;
        }
    }

    pub fn momentum(&self, phase: usize) -> &VectorDim<DIM> {
        &self.momentum[phase]
    }

    pub fn update_mass_pressure(&mut self, phase: usize, mass_pressure: f64) {
        self.pressure[phase] += mass_pressure;
    }

    pub fn assign_pressure(&mut self, phase: usize, mass_pressure: f64) {
        if self.mass[phase] > std::f64::EPSILON {
            self.pressure[phase] = mass_pressure / self.mass[phase];
        }
    }

    pub fn pressure(&self, phase: usize) -> f64 {
        self.pressure[phase]
    }

    /// Compute velocity from momentum: v = p / m.
    pub fn compute_velocity(&mut self) {
        for phase in 0..NPHASES {
            if self.mass[phase] > std::f64::EPSILON {
                self.velocity[phase] = self.momentum[phase] / self.mass[phase];
            }
        }
    }

    pub fn velocity(&self, phase: usize) -> &VectorDim<DIM> {
        &self.velocity[phase]
    }

    pub fn update_acceleration(&mut self, update: bool, phase: usize, acceleration: &VectorDim<DIM>) {
        if update {
            self.acceleration[phase] += acceleration;
        } else {
            self.acceleration[phase] = *acceleration;
        }
    }

    pub fn acceleration(&self, phase: usize) -> &VectorDim<DIM> {
        &self.acceleration[phase]
    }

    /// Compute acceleration and velocity from forces.
    pub fn compute_acceleration_velocity(&mut self, phase: usize, dt: f64) -> bool {
        if self.mass[phase] > std::f64::EPSILON {
            let force = self.internal_force[phase] + self.external_force[phase];
            self.acceleration[phase] = force / self.mass[phase];
            self.velocity[phase] += self.acceleration[phase] * dt;
            true
        } else {
            false
        }
    }

    /// Compute acceleration with Cundall damping.
    pub fn compute_acceleration_velocity_cundall(
        &mut self,
        phase: usize,
        dt: f64,
        damping_factor: f64,
    ) -> bool {
        if self.mass[phase] > std::f64::EPSILON {
            let force = self.internal_force[phase] + self.external_force[phase];

            // Cundall damping: reduce unbalanced force
            let mut damped_force = force;
            let force_norm = force.norm();
            if force_norm > std::f64::EPSILON {
                for d in 0..DIM {
                    let vel_sign = if self.velocity[phase][d] >= 0.0 { 1.0 } else { -1.0 };
                    damped_force[d] -= damping_factor * force[d].abs() * vel_sign;
                }
            }

            self.acceleration[phase] = damped_force / self.mass[phase];
            self.velocity[phase] += self.acceleration[phase] * dt;
            true
        } else {
            false
        }
    }

    pub fn assign_velocity_constraint(&mut self, dir: usize, velocity: f64) -> bool {
        if dir < DIM {
            self.velocity_constraints.insert(dir, velocity);
            true
        } else {
            false
        }
    }

    /// Apply velocity constraints.
    pub fn apply_velocity_constraints(&mut self) {
        for (&dir, &vel) in &self.velocity_constraints {
            for phase in 0..NPHASES {
                self.velocity[phase][dir] = vel;
                self.momentum[phase][dir] = self.mass[phase] * vel;
                self.acceleration[phase][dir] = 0.0;
            }
        }
    }

    pub fn assign_friction_constraint(&mut self, dir: usize, sign_n: i32, friction: f64) -> bool {
        if dir < DIM {
            self.friction_constraint = Some((dir, sign_n, friction));
            true
        } else {
            false
        }
    }

    /// Apply friction constraints.
    pub fn apply_friction_constraints(&mut self, dt: f64) {
        if let Some((dir, sign_n, friction)) = self.friction_constraint {
            for phase in 0..NPHASES {
                if self.mass[phase] > std::f64::EPSILON {
                    let normal_force = self.internal_force[phase][dir]
                        + self.external_force[phase][dir];

                    let is_contact = (sign_n > 0 && normal_force > 0.0)
                        || (sign_n < 0 && normal_force < 0.0);

                    if is_contact {
                        // Apply friction in tangential directions
                        let fn_mag = normal_force.abs();
                        for d in 0..DIM {
                            if d != dir {
                                let ft = self.internal_force[phase][d]
                                    + self.external_force[phase][d];
                                let ft_max = friction * fn_mag;
                                if ft.abs() > ft_max {
                                    let ft_applied = ft.signum() * ft_max;
                                    self.velocity[phase][d] +=
                                        (ft_applied - ft) / self.mass[phase] * dt;
                                }
                            }
                        }
                        // Zero normal
                        self.velocity[phase][dir] = 0.0;
                        self.acceleration[phase][dir] = 0.0;
                    }
                }
            }
        }
    }

    pub fn assign_rotation_matrix(&mut self, rotation_matrix: SMatrix<f64, DIM, DIM>) {
        self.rotation_matrix = rotation_matrix;
        self.generic_boundary_constraints = true;
    }

    pub fn assign_concentrated_force(
        &mut self,
        phase: usize,
        direction: usize,
        force: f64,
        function: Option<Arc<dyn Function>>,
    ) -> bool {
        if direction < DIM {
            self.concentrated_force[phase][direction] = force;
            self.force_function = function;
            true
        } else {
            false
        }
    }

    pub fn apply_concentrated_force(&mut self, phase: usize, current_time: f64) {
        let scale = match &self.force_function {
            Some(f) => f.value(current_time),
            None => 1.0,
        };
        self.external_force[phase] += self.concentrated_force[phase] * scale;
    }

    pub fn append_material_id(&mut self, id: u32) {
        self.material_ids.insert(id);
    }

    pub fn material_ids(&self) -> &BTreeSet<u32> {
        &self.material_ids
    }

    pub fn ghost_id(&self) -> Index {
        self.ghost_id
    }

    pub fn set_ghost_id(&mut self, gid: Index) {
        self.ghost_id = gid;
    }

    #[cfg(feature = "mpi")]
    pub fn mpi_rank(&mut self, rank: u32) -> bool {
        self.mpi_ranks.insert(rank)
    }

    #[cfg(feature = "mpi")]
    pub fn mpi_ranks(&self) -> &BTreeSet<u32> {
        &self.mpi_ranks
    }

    #[cfg(feature = "mpi")]
    pub fn clear_mpi_ranks(&mut self) {
        self.mpi_ranks.clear();
    }
}
