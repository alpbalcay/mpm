//! Spatial cell containing particles and referencing element shape functions.

use mpm_core::{Index, VectorDim};
use mpm_elements::element::Element;
use mpm_elements::quadrature::Quadrature;
use nalgebra::DMatrix;
use std::collections::BTreeSet;
use std::sync::Arc;

/// A cell in the background mesh that contains particles.
pub struct Cell<const DIM: usize> {
    id: Index,
    rank: u32,
    previous_mpirank: u32,
    isoparametric: bool,
    nnodes: usize,
    volume: f64,
    centroid: VectorDim<DIM>,
    mean_length: f64,
    particle_ids: Vec<Index>,
    nglobal_particles: u32,
    node_ids: Vec<Index>,
    nodal_coordinates: DMatrix<f64>,
    neighbours: BTreeSet<Index>,
    element: Arc<dyn Element<DIM>>,
    quadrature: Option<Box<dyn Quadrature<DIM>>>,
    dn_dx_centroid: DMatrix<f64>,
}

impl<const DIM: usize> Cell<DIM> {
    pub fn new(
        id: Index,
        nnodes: usize,
        element: Arc<dyn Element<DIM>>,
        isoparametric: bool,
    ) -> Self {
        Self {
            id,
            rank: 0,
            previous_mpirank: 0,
            isoparametric,
            nnodes,
            volume: 0.0,
            centroid: VectorDim::<DIM>::zeros(),
            mean_length: 0.0,
            particle_ids: Vec::new(),
            nglobal_particles: 0,
            node_ids: Vec::with_capacity(nnodes),
            nodal_coordinates: DMatrix::zeros(nnodes, DIM),
            neighbours: BTreeSet::new(),
            element,
            quadrature: None,
            dn_dx_centroid: DMatrix::zeros(0, 0),
        }
    }

    pub fn id(&self) -> Index {
        self.id
    }

    pub fn is_initialised(&self) -> bool {
        self.node_ids.len() == self.nnodes
    }

    pub fn assign_quadrature(&mut self, nquadratures: usize) {
        self.quadrature = Some(self.element.quadrature(nquadratures));
    }

    /// Generate quadrature points in physical coordinates.
    pub fn generate_points(&self) -> Vec<VectorDim<DIM>> {
        let quad = match &self.quadrature {
            Some(q) => q,
            None => return Vec::new(),
        };

        let points = quad.points();
        let npts = quad.npoints();
        let zero_vec = VectorDim::<DIM>::zeros();
        let mut physical_points = Vec::with_capacity(npts);

        for i in 0..npts {
            let mut xi = VectorDim::<DIM>::zeros();
            for d in 0..DIM {
                xi[d] = points[(i, d)];
            }
            let sf = self.element.shapefn(&xi, &zero_vec, &zero_vec);
            let mut point = VectorDim::<DIM>::zeros();
            for n in 0..self.nnodes.min(sf.len()) {
                for d in 0..DIM {
                    point[d] += sf[n] * self.nodal_coordinates[(n, d)];
                }
            }
            physical_points.push(point);
        }
        physical_points
    }

    pub fn nparticles(&self) -> usize {
        self.particle_ids.len()
    }

    pub fn set_nglobal_particles(&mut self, n: u32) {
        self.nglobal_particles = n;
    }

    pub fn nglobal_particles(&self) -> u32 {
        self.nglobal_particles
    }

    pub fn status(&self) -> bool {
        !self.particle_ids.is_empty()
    }

    pub fn particle_ids(&self) -> &[Index] {
        &self.particle_ids
    }

    pub fn nnodes(&self) -> usize {
        self.node_ids.len()
    }

    pub fn node_ids(&self) -> &[Index] {
        &self.node_ids
    }

    pub fn nfunctions(&self) -> usize {
        self.element.nfunctions()
    }

    pub fn add_node(&mut self, local_id: usize, node_id: Index, coord: &VectorDim<DIM>) -> bool {
        if local_id < self.nnodes {
            if self.node_ids.len() <= local_id {
                self.node_ids.resize(local_id + 1, 0);
            }
            self.node_ids[local_id] = node_id;
            for d in 0..DIM {
                self.nodal_coordinates[(local_id, d)] = coord[d];
            }
            true
        } else {
            false
        }
    }

    pub fn add_neighbour(&mut self, neighbour_id: Index) -> bool {
        if neighbour_id != self.id {
            self.neighbours.insert(neighbour_id);
            true
        } else {
            false
        }
    }

    pub fn nneighbours(&self) -> usize {
        self.neighbours.len()
    }

    pub fn neighbours(&self) -> &BTreeSet<Index> {
        &self.neighbours
    }

    pub fn add_particle_id(&mut self, id: Index) -> bool {
        self.particle_ids.push(id);
        true
    }

    pub fn remove_particle_id(&mut self, id: Index) {
        self.particle_ids.retain(|&pid| pid != id);
    }

    pub fn clear_particle_ids(&mut self) {
        self.particle_ids.clear();
    }

    /// Compute cell volume from nodal coordinates.
    pub fn compute_volume(&mut self) {
        self.volume = self.element.compute_volume(&self.nodal_coordinates);
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Compute cell centroid from nodal coordinates.
    pub fn compute_centroid(&mut self) {
        let mut centroid = VectorDim::<DIM>::zeros();
        let n = self.node_ids.len() as f64;
        for i in 0..self.node_ids.len() {
            for d in 0..DIM {
                centroid[d] += self.nodal_coordinates[(i, d)];
            }
        }
        self.centroid = centroid / n;
    }

    pub fn centroid(&self) -> &VectorDim<DIM> {
        &self.centroid
    }

    /// Compute mean element length.
    pub fn compute_mean_length(&mut self) {
        self.mean_length = self.volume.powf(1.0 / DIM as f64);
    }

    pub fn mean_length(&self) -> f64 {
        self.mean_length
    }

    pub fn nodal_coordinates(&self) -> &DMatrix<f64> {
        &self.nodal_coordinates
    }

    pub fn dn_dx_centroid(&self) -> &DMatrix<f64> {
        &self.dn_dx_centroid
    }

    /// Compute dn_dx at cell centroid.
    pub fn compute_dn_dx_centroid(&mut self) {
        let xi = VectorDim::<DIM>::zeros();
        let zero = VectorDim::<DIM>::zeros();
        self.dn_dx_centroid = self.element.dn_dx(
            &xi,
            &self.nodal_coordinates,
            &zero,
            &zero,
        );
    }

    pub fn element_ptr(&self) -> &Arc<dyn Element<DIM>> {
        &self.element
    }

    /// Check if a point is inside this cell using Newton-Raphson inverse mapping.
    pub fn is_point_in_cell(&self, point: &VectorDim<DIM>) -> Option<VectorDim<DIM>> {
        if self.element.isvalid_natural_coordinates_analytical() {
            let xi = self
                .element
                .natural_coordinates_analytical(point, &self.nodal_coordinates);
            if self.is_valid_natural_coordinate(&xi) {
                return Some(xi);
            }
            return None;
        }

        // Newton-Raphson
        let mut xi = self.transform_real_to_unit_cell(point);
        if self.is_valid_natural_coordinate(&xi) {
            Some(xi)
        } else {
            None
        }
    }

    /// Transform real coordinates to natural coordinates using Newton-Raphson.
    pub fn transform_real_to_unit_cell(&self, point: &VectorDim<DIM>) -> VectorDim<DIM> {
        let zero = VectorDim::<DIM>::zeros();
        let mut xi = VectorDim::<DIM>::zeros();

        for _ in 0..100 {
            let sf = self.element.shapefn(&xi, &zero, &zero);
            let mut residual = VectorDim::<DIM>::zeros();
            for d in 0..DIM {
                for n in 0..sf.len().min(self.nnodes) {
                    residual[d] += sf[n] * self.nodal_coordinates[(n, d)];
                }
                residual[d] -= point[d];
            }

            if residual.norm() < 1.0e-12 {
                break;
            }

            let jac = self.element.jacobian(
                &xi,
                &self.nodal_coordinates,
                &zero,
                &zero,
            );
            if let Some(jac_inv) = jac.try_inverse() {
                xi -= jac_inv * residual;
            } else {
                break;
            }
        }
        xi
    }

    fn is_valid_natural_coordinate(&self, xi: &VectorDim<DIM>) -> bool {
        let tol = 1.0 + 1.0e-6;
        let len = self.element.unit_element_length();
        for d in 0..DIM {
            if xi[d].abs() > tol * len / 2.0 {
                return false;
            }
        }
        true
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn set_rank(&mut self, rank: u32) {
        self.previous_mpirank = self.rank;
        self.rank = rank;
    }

    pub fn previous_mpirank(&self) -> u32 {
        self.previous_mpirank
    }

    /// Initialise cell: compute volume, centroid, mean_length, dn_dx.
    pub fn initialise(&mut self) -> bool {
        if !self.is_initialised() {
            return false;
        }
        self.compute_volume();
        self.compute_centroid();
        self.compute_mean_length();
        self.compute_dn_dx_centroid();
        true
    }
}
