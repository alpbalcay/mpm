//! Mesh container managing nodes, cells, and particles.

use crate::cell::Cell;
use crate::node::Node;
use crate::nodal_properties::NodalProperties;
use crate::particle::Particle;
use mpm_core::{Index, VectorDim};
use mpm_elements::element::Element;
use mpm_materials::material::Material;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// The top-level mesh container for MPM.
pub struct Mesh<const DIM: usize> {
    id: u32,
    isoparametric: bool,

    // Domain objects
    nodes: Vec<Arc<RwLock<Node<DIM>>>>,
    cells: Vec<Arc<RwLock<Cell<DIM>>>>,
    particles: Vec<Arc<RwLock<Particle<DIM>>>>,

    // Fast lookup by ID
    node_map: HashMap<Index, usize>,
    cell_map: HashMap<Index, usize>,
    particle_map: HashMap<Index, usize>,

    // Sets
    node_sets: HashMap<u32, Vec<Index>>,
    cell_sets: HashMap<u32, Vec<Index>>,
    particle_sets: HashMap<u32, Vec<Index>>,

    // Materials
    materials: HashMap<u32, Arc<dyn Material<DIM>>>,

    // Nodal properties (for multimaterial contact)
    nodal_properties: Arc<NodalProperties>,
}

impl<const DIM: usize> Mesh<DIM> {
    pub fn new(id: u32, isoparametric: bool) -> Self {
        Self {
            id,
            isoparametric,
            nodes: Vec::new(),
            cells: Vec::new(),
            particles: Vec::new(),
            node_map: HashMap::new(),
            cell_map: HashMap::new(),
            particle_map: HashMap::new(),
            node_sets: HashMap::new(),
            cell_sets: HashMap::new(),
            particle_sets: HashMap::new(),
            materials: HashMap::new(),
            nodal_properties: Arc::new(NodalProperties::new()),
        }
    }

    pub fn id(&self) -> u32 { self.id }
    pub fn is_isoparametric(&self) -> bool { self.isoparametric }
    pub fn nnodes(&self) -> usize { self.nodes.len() }
    pub fn ncells(&self) -> usize { self.cells.len() }
    pub fn nparticles(&self) -> usize { self.particles.len() }

    /// Create nodes from coordinate vectors.
    pub fn create_nodes(
        &mut self,
        gnid: Index,
        coordinates: &[VectorDim<DIM>],
    ) -> bool {
        for (i, coord) in coordinates.iter().enumerate() {
            let nid = gnid + i as u64;
            let node = Arc::new(RwLock::new(Node::new(nid, *coord)));
            let idx = self.nodes.len();
            self.nodes.push(node);
            self.node_map.insert(nid, idx);
        }
        true
    }

    /// Add a single node.
    pub fn add_node(&mut self, node: Arc<RwLock<Node<DIM>>>) -> bool {
        let nid = node.read().unwrap().id();
        if self.node_map.contains_key(&nid) {
            return false;
        }
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.node_map.insert(nid, idx);
        true
    }

    /// Get a node by ID.
    pub fn node(&self, node_id: Index) -> Option<Arc<RwLock<Node<DIM>>>> {
        self.node_map
            .get(&node_id)
            .map(|&idx| self.nodes[idx].clone())
    }

    /// Create cells from connectivity lists.
    pub fn create_cells(
        &mut self,
        gnid: Index,
        element: Arc<dyn Element<DIM>>,
        connectivity: &[Vec<Index>],
    ) -> bool {
        for (i, cell_nodes) in connectivity.iter().enumerate() {
            let cid = gnid + i as u64;
            let nnodes = cell_nodes.len();
            let mut cell = Cell::new(cid, nnodes, element.clone(), self.isoparametric);

            for (local_id, &node_id) in cell_nodes.iter().enumerate() {
                if let Some(node) = self.node(node_id) {
                    let coord = *node.read().unwrap().coordinates();
                    cell.add_node(local_id, node_id, &coord);
                }
            }
            cell.initialise();

            let cell = Arc::new(RwLock::new(cell));
            let idx = self.cells.len();
            self.cells.push(cell);
            self.cell_map.insert(cid, idx);
        }
        true
    }

    /// Get a cell by ID.
    pub fn cell(&self, cell_id: Index) -> Option<Arc<RwLock<Cell<DIM>>>> {
        self.cell_map
            .get(&cell_id)
            .map(|&idx| self.cells[idx].clone())
    }

    /// Create particles from coordinates.
    pub fn create_particles(
        &mut self,
        coordinates: &[VectorDim<DIM>],
        material_ids: &[u32],
    ) -> bool {
        for (i, coord) in coordinates.iter().enumerate() {
            let pid = i as Index;
            let mut particle = Particle::new(pid, *coord);

            if i < material_ids.len() {
                if let Some(material) = self.materials.get(&material_ids[i]) {
                    particle.assign_material(material.clone());
                }
            }

            let idx = self.particles.len();
            self.particles.push(Arc::new(RwLock::new(particle)));
            self.particle_map.insert(pid, idx);
        }
        true
    }

    /// Get a particle by ID.
    pub fn particle(&self, particle_id: Index) -> Option<Arc<RwLock<Particle<DIM>>>> {
        self.particle_map
            .get(&particle_id)
            .map(|&idx| self.particles[idx].clone())
    }

    /// Add a material.
    pub fn add_material(&mut self, id: u32, material: Arc<dyn Material<DIM>>) {
        self.materials.insert(id, material);
    }

    /// Get materials map.
    pub fn materials(&self) -> &HashMap<u32, Arc<dyn Material<DIM>>> {
        &self.materials
    }

    /// Locate particles in cells.
    pub fn locate_particles(&self) -> Vec<Index> {
        let mut unlocated = Vec::new();
        for particle in &self.particles {
            let mut p = particle.write().unwrap();
            let coord = *p.coordinates();
            let mut found = false;

            for cell in &self.cells {
                let c = cell.read().unwrap();
                if let Some(xi) = c.is_point_in_cell(&coord) {
                    p.assign_cell_id(c.id());
                    p.set_xi(xi);
                    p.set_node_ids(c.node_ids().to_vec());
                    drop(c);
                    cell.write().unwrap().add_particle_id(p.id());
                    found = true;
                    break;
                }
            }

            if !found {
                unlocated.push(p.id());
                p.assign_status(false);
            }
        }
        unlocated
    }

    /// Find active nodes (nodes with particles in their cells).
    pub fn find_active_nodes(&self) {
        // Reset all nodes
        for node in &self.nodes {
            node.write().unwrap().assign_status(false);
        }

        // Activate nodes in cells with particles
        for cell in &self.cells {
            let c = cell.read().unwrap();
            if c.status() {
                for &node_id in c.node_ids() {
                    if let Some(node) = self.node(node_id) {
                        node.write().unwrap().assign_status(true);
                    }
                }
            }
        }
    }

    /// Iterate over all particles.
    pub fn iterate_over_particles<F: Fn(&Arc<RwLock<Particle<DIM>>>)>(&self, f: F) {
        for particle in &self.particles {
            f(particle);
        }
    }

    /// Iterate over all nodes.
    pub fn iterate_over_nodes<F: Fn(&Arc<RwLock<Node<DIM>>>)>(&self, f: F) {
        for node in &self.nodes {
            f(node);
        }
    }

    /// Iterate over active nodes.
    pub fn iterate_over_active_nodes<F: Fn(&Arc<RwLock<Node<DIM>>>)>(&self, f: F) {
        for node in &self.nodes {
            if node.read().unwrap().status() {
                f(node);
            }
        }
    }

    /// Iterate over all cells.
    pub fn iterate_over_cells<F: Fn(&Arc<RwLock<Cell<DIM>>>)>(&self, f: F) {
        for cell in &self.cells {
            f(cell);
        }
    }

    /// Create node sets.
    pub fn create_node_sets(&mut self, sets: HashMap<u32, Vec<Index>>) {
        self.node_sets = sets;
    }

    /// Create cell sets.
    pub fn create_cell_sets(&mut self, sets: HashMap<u32, Vec<Index>>) {
        self.cell_sets = sets;
    }

    /// Create particle sets.
    pub fn create_particle_sets(&mut self, sets: HashMap<u32, Vec<Index>>) {
        self.particle_sets = sets;
    }

    pub fn node_set(&self, set_id: u32) -> Option<&Vec<Index>> {
        self.node_sets.get(&set_id)
    }

    pub fn cell_set(&self, set_id: u32) -> Option<&Vec<Index>> {
        self.cell_sets.get(&set_id)
    }

    pub fn particle_set(&self, set_id: u32) -> Option<&Vec<Index>> {
        self.particle_sets.get(&set_id)
    }

    /// Get particle coordinates for output.
    pub fn particle_coordinates(&self) -> Vec<[f64; 3]> {
        self.particles
            .iter()
            .map(|p| {
                let p = p.read().unwrap();
                let c = p.coordinates();
                let mut coords = [0.0; 3];
                for d in 0..DIM.min(3) {
                    coords[d] = c[d];
                }
                coords
            })
            .collect()
    }

    /// Get particle scalar data.
    pub fn particles_scalar_data(&self, attribute: &str) -> Vec<f64> {
        self.particles
            .iter()
            .map(|p| p.read().unwrap().scalar_data(attribute))
            .collect()
    }

    /// Get all nodes (for iteration).
    pub fn nodes_ref(&self) -> &[Arc<RwLock<Node<DIM>>>] {
        &self.nodes
    }

    /// Get all cells.
    pub fn cells_ref(&self) -> &[Arc<RwLock<Cell<DIM>>>] {
        &self.cells
    }

    /// Get all particles.
    pub fn particles_ref(&self) -> &[Arc<RwLock<Particle<DIM>>>] {
        &self.particles
    }

    /// Find cell neighbours based on shared nodes.
    pub fn find_cell_neighbours(&self) {
        // Build node-to-cells map
        let mut node_cells: HashMap<Index, Vec<Index>> = HashMap::new();
        for cell in &self.cells {
            let c = cell.read().unwrap();
            for &nid in c.node_ids() {
                node_cells.entry(nid).or_default().push(c.id());
            }
        }

        // Assign neighbours
        for cell in &self.cells {
            let cid = cell.read().unwrap().id();
            let node_ids: Vec<Index> = cell.read().unwrap().node_ids().to_vec();
            let mut neighbours = std::collections::BTreeSet::new();

            for nid in &node_ids {
                if let Some(cells) = node_cells.get(nid) {
                    for &other_cid in cells {
                        if other_cid != cid {
                            neighbours.insert(other_cid);
                        }
                    }
                }
            }

            let mut c = cell.write().unwrap();
            for nid in neighbours {
                c.add_neighbour(nid);
            }
        }
    }

    /// Initialise all nodes.
    pub fn initialise_nodes(&self) {
        for node in &self.nodes {
            node.write().unwrap().initialise();
        }
    }

    /// Compute shape functions for all particles.
    pub fn compute_particle_shapefns(&self) {
        for particle in &self.particles {
            let mut p = particle.write().unwrap();
            if let Some(cell_id) = p.cell_id() {
                if let Some(cell) = self.cell(cell_id) {
                    let c = cell.read().unwrap();
                    let elem = c.element_ptr();
                    let xi = *p.reference_location();
                    let zero = VectorDim::<DIM>::zeros();
                    let psize = *p.natural_size();

                    let sf = elem.shapefn(&xi, &psize, &zero);
                    let dn = elem.dn_dx(&xi, c.nodal_coordinates(), &psize, &zero);
                    let dn_c = c.dn_dx_centroid().clone();

                    p.set_shapefn_data(sf, dn, dn_c);
                }
            }
        }
    }
}
