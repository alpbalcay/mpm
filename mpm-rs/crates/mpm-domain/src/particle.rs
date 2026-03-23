//! Material point (particle) for MPM.

use mpm_core::{DenseMap, Index, Vector6d, VectorDim};
use mpm_materials::material::{Material, ParticleData};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

/// A material point that carries physical state.
pub struct Particle<const DIM: usize> {
    id: Index,
    coordinates: VectorDim<DIM>,
    cell_id: Option<Index>,
    xi: VectorDim<DIM>,
    status: bool,

    // Physical properties
    mass: f64,
    mass_density: f64,
    volume: f64,
    size: VectorDim<DIM>,
    natural_size: VectorDim<DIM>,

    // Kinematic state
    velocity: VectorDim<DIM>,
    displacement: VectorDim<DIM>,

    // Stress/strain state (Voigt notation)
    stress: Vector6d,
    strain: Vector6d,
    dstrain: Vector6d,
    strain_rate: Vector6d,
    dvolumetric_strain: f64,
    volumetric_strain_centroid: f64,

    // Traction
    set_traction: bool,
    traction: VectorDim<DIM>,

    // Shape function data (cached)
    shapefn: DVector<f64>,
    dn_dx: DMatrix<f64>,
    dn_dx_centroid: DMatrix<f64>,

    // Material
    material: Option<Arc<dyn Material<DIM>>>,
    material_id: Option<u32>,
    state_variables: DenseMap,

    // Constraints
    velocity_constraints: std::collections::BTreeMap<usize, f64>,

    // Neighbours
    neighbours: Vec<Index>,

    // Node IDs (cached from cell)
    node_ids: Vec<Index>,
}

impl<const DIM: usize> Particle<DIM> {
    pub fn new(id: Index, coordinates: VectorDim<DIM>) -> Self {
        Self {
            id,
            coordinates,
            cell_id: None,
            xi: VectorDim::<DIM>::zeros(),
            status: true,
            mass: 0.0,
            mass_density: 0.0,
            volume: 0.0,
            size: VectorDim::<DIM>::from_element(1.0),
            natural_size: VectorDim::<DIM>::from_element(1.0),
            velocity: VectorDim::<DIM>::zeros(),
            displacement: VectorDim::<DIM>::zeros(),
            stress: Vector6d::zeros(),
            strain: Vector6d::zeros(),
            dstrain: Vector6d::zeros(),
            strain_rate: Vector6d::zeros(),
            dvolumetric_strain: 0.0,
            volumetric_strain_centroid: 0.0,
            set_traction: false,
            traction: VectorDim::<DIM>::zeros(),
            shapefn: DVector::zeros(0),
            dn_dx: DMatrix::zeros(0, 0),
            dn_dx_centroid: DMatrix::zeros(0, 0),
            material: None,
            material_id: None,
            state_variables: DenseMap::new(),
            velocity_constraints: std::collections::BTreeMap::new(),
            neighbours: Vec::new(),
            node_ids: Vec::new(),
        }
    }

    // --- Accessors ---

    pub fn id(&self) -> Index { self.id }
    pub fn coordinates(&self) -> &VectorDim<DIM> { &self.coordinates }
    pub fn assign_coordinates(&mut self, coord: VectorDim<DIM>) { self.coordinates = coord; }
    pub fn cell_id(&self) -> Option<Index> { self.cell_id }
    pub fn assign_cell_id(&mut self, cell_id: Index) { self.cell_id = Some(cell_id); }
    pub fn reference_location(&self) -> &VectorDim<DIM> { &self.xi }
    pub fn status(&self) -> bool { self.status }
    pub fn assign_status(&mut self, status: bool) { self.status = status; }

    pub fn mass(&self) -> f64 { self.mass }
    pub fn assign_mass(&mut self, mass: f64) { self.mass = mass; }
    pub fn mass_density(&self) -> f64 { self.mass_density }
    pub fn volume(&self) -> f64 { self.volume }
    pub fn assign_volume(&mut self, volume: f64) { self.volume = volume; }
    pub fn natural_size(&self) -> &VectorDim<DIM> { &self.natural_size }

    pub fn velocity(&self) -> &VectorDim<DIM> { &self.velocity }
    pub fn assign_velocity(&mut self, velocity: VectorDim<DIM>) { self.velocity = velocity; }
    pub fn displacement(&self) -> &VectorDim<DIM> { &self.displacement }

    pub fn stress(&self) -> &Vector6d { &self.stress }
    pub fn strain(&self) -> &Vector6d { &self.strain }
    pub fn strain_rate_value(&self) -> &Vector6d { &self.strain_rate }
    pub fn dvolumetric_strain(&self) -> f64 { self.dvolumetric_strain }
    pub fn volumetric_strain_centroid(&self) -> f64 { self.volumetric_strain_centroid }

    pub fn initial_stress(&mut self, stress: Vector6d) { self.stress = stress; }

    pub fn traction_value(&self) -> &VectorDim<DIM> { &self.traction }
    pub fn assign_traction(&mut self, direction: usize, traction: f64) -> bool {
        if direction < DIM {
            self.set_traction = true;
            self.traction[direction] = traction;
            true
        } else {
            false
        }
    }

    pub fn node_ids(&self) -> &[Index] { &self.node_ids }
    pub fn set_node_ids(&mut self, ids: Vec<Index>) { self.node_ids = ids; }

    pub fn shapefn(&self) -> &DVector<f64> { &self.shapefn }
    pub fn dn_dx(&self) -> &DMatrix<f64> { &self.dn_dx }

    pub fn material_id(&self) -> Option<u32> { self.material_id }

    pub fn state_variables(&self) -> &DenseMap { &self.state_variables }
    pub fn state_variables_mut(&mut self) -> &mut DenseMap { &mut self.state_variables }

    pub fn state_variable(&self, var: &str) -> Option<f64> {
        self.state_variables.get(var).copied()
    }

    pub fn assign_material_state_vars(&mut self, state_vars: DenseMap) {
        self.state_variables = state_vars;
    }

    // --- Material ---

    pub fn assign_material(&mut self, material: Arc<dyn Material<DIM>>) -> bool {
        self.material_id = Some(material.id() as u32);
        self.mass_density = material.density();
        self.state_variables = material.initialise_state_variables();
        self.material = Some(material);
        true
    }

    // --- Computation Methods ---

    /// Initialise particle for a new step.
    pub fn initialise(&mut self) {
        // No per-step reset needed for most fields
    }

    pub fn set_xi(&mut self, xi: VectorDim<DIM>) {
        self.xi = xi;
    }

    /// Set cached shape functions and gradients.
    pub fn set_shapefn_data(
        &mut self,
        shapefn: DVector<f64>,
        dn_dx: DMatrix<f64>,
        dn_dx_centroid: DMatrix<f64>,
    ) {
        self.shapefn = shapefn;
        self.dn_dx = dn_dx;
        self.dn_dx_centroid = dn_dx_centroid;
    }

    /// Compute volume from mass and density.
    pub fn compute_volume(&mut self) {
        if self.mass_density > 0.0 {
            self.volume = self.mass / self.mass_density;
        }
    }

    /// Compute mass from volume and density.
    pub fn compute_mass(&mut self) {
        self.mass = self.volume * self.mass_density;
    }

    /// Compute strain increment from nodal velocities.
    pub fn compute_strain_from_velocities(&mut self, nodal_velocities: &[VectorDim<DIM>], dt: f64) {
        let nfn = self.dn_dx.nrows();
        let mut strain_rate = Vector6d::zeros();

        for i in 0..nfn.min(nodal_velocities.len()) {
            if DIM >= 2 {
                strain_rate[0] += self.dn_dx[(i, 0)] * nodal_velocities[i][0];
                strain_rate[1] += self.dn_dx[(i, 1)] * nodal_velocities[i][1];
                strain_rate[3] += self.dn_dx[(i, 1)] * nodal_velocities[i][0]
                    + self.dn_dx[(i, 0)] * nodal_velocities[i][1];
            }
            if DIM == 3 {
                strain_rate[2] += self.dn_dx[(i, 2)] * nodal_velocities[i][2];
                strain_rate[4] += self.dn_dx[(i, 2)] * nodal_velocities[i][1]
                    + self.dn_dx[(i, 1)] * nodal_velocities[i][2];
                strain_rate[5] += self.dn_dx[(i, 2)] * nodal_velocities[i][0]
                    + self.dn_dx[(i, 0)] * nodal_velocities[i][2];
            }
        }

        self.strain_rate = strain_rate;
        self.dstrain = strain_rate * dt;
        self.strain += self.dstrain;

        // Volumetric strain
        self.dvolumetric_strain = self.dstrain[0] + self.dstrain[1] + self.dstrain[2];

        // Centroid volumetric strain
        let nfn_c = self.dn_dx_centroid.nrows();
        let mut dvol_centroid = 0.0;
        for i in 0..nfn_c.min(nodal_velocities.len()) {
            for d in 0..DIM {
                dvol_centroid += self.dn_dx_centroid[(i, d)] * nodal_velocities[i][d] * dt;
            }
        }
        self.volumetric_strain_centroid += dvol_centroid;
    }

    /// Compute stress using the material constitutive model.
    pub fn compute_stress(&mut self) {
        if let Some(ref material) = self.material {
            // Create a temporary ParticleData snapshot to avoid borrow conflict
            let snapshot = ParticleSnapshot {
                strain_rate: self.strain_rate,
                pressure: -(self.stress[0] + self.stress[1] + self.stress[2]) / 3.0,
                previous_stress: self.stress,
            };
            let new_stress = material.compute_stress(
                &self.stress,
                &self.dstrain,
                &snapshot,
                &mut self.state_variables,
            );
            self.stress = new_stress;
        }
    }

    /// Update volume based on volumetric strain.
    pub fn update_volume(&mut self) {
        self.volume *= (1.0 + self.dvolumetric_strain);
        if self.volume > 0.0 {
            self.mass_density = self.mass / self.volume;
        }
    }

    /// Compute updated position from nodal velocities.
    pub fn compute_updated_position_from_velocities(
        &mut self,
        nodal_velocities: &[VectorDim<DIM>],
        dt: f64,
        velocity_update: bool,
    ) {
        let nfn = self.shapefn.len();
        let mut displacement_inc = VectorDim::<DIM>::zeros();
        let mut velocity_inc = VectorDim::<DIM>::zeros();

        for i in 0..nfn.min(nodal_velocities.len()) {
            displacement_inc += nodal_velocities[i] * self.shapefn[i] * dt;
            velocity_inc += nodal_velocities[i] * self.shapefn[i];
        }

        self.coordinates += displacement_inc;
        self.displacement += displacement_inc;

        if velocity_update {
            self.velocity = velocity_inc;
        } else {
            // PIC/FLIP blend - use FLIP
            self.velocity += velocity_inc;
        }

        // Apply velocity constraints
        for (&dir, &vel) in &self.velocity_constraints {
            self.velocity[dir] = vel;
        }
    }

    pub fn apply_particle_velocity_constraints(&mut self, dir: usize, velocity: f64) {
        self.velocity_constraints.insert(dir, velocity);
    }

    pub fn assign_neighbours(&mut self, neighbours: Vec<Index>) {
        self.neighbours = neighbours;
    }

    pub fn neighbours(&self) -> &[Index] {
        &self.neighbours
    }

    pub fn nneighbours(&self) -> usize {
        self.neighbours.len()
    }

    /// Scalar data accessor for output.
    pub fn scalar_data(&self, property: &str) -> f64 {
        match property {
            "mass" => self.mass,
            "volume" => self.volume,
            "mass_density" => self.mass_density,
            "pressure" => -(self.stress[0] + self.stress[1] + self.stress[2]) / 3.0,
            "nneighbours" => self.neighbours.len() as f64,
            _ => {
                // Check state variables
                self.state_variables.get(property).copied().unwrap_or(0.0)
            }
        }
    }

    /// Vector data accessor for output.
    pub fn vector_data(&self, property: &str) -> VectorDim<DIM> {
        match property {
            "velocities" => self.velocity,
            "displacements" => self.displacement,
            _ => VectorDim::<DIM>::zeros(),
        }
    }

    /// Tensor data accessor for output.
    pub fn tensor_data(&self, property: &str) -> Vector6d {
        match property {
            "stresses" => self.stress,
            "strains" => self.strain,
            "strain_rates" => self.strain_rate,
            _ => Vector6d::zeros(),
        }
    }

    pub fn particle_type(&self) -> &str {
        if DIM == 2 { "P2D" } else { "P3D" }
    }
}

/// Snapshot of particle data for passing to material models without borrow conflicts.
struct ParticleSnapshot {
    strain_rate: Vector6d,
    pressure: f64,
    previous_stress: Vector6d,
}

impl ParticleData for ParticleSnapshot {
    fn strain_rate(&self) -> Vector6d {
        self.strain_rate
    }

    fn pressure(&self) -> f64 {
        self.pressure
    }

    fn previous_stress(&self) -> Vector6d {
        self.previous_stress
    }
}

/// Implement ParticleData trait so materials can read particle state.
impl<const DIM: usize> ParticleData for Particle<DIM> {
    fn strain_rate(&self) -> Vector6d {
        self.strain_rate
    }

    fn pressure(&self) -> f64 {
        -(self.stress[0] + self.stress[1] + self.stress[2]) / 3.0
    }

    fn previous_stress(&self) -> Vector6d {
        self.stress
    }
}
