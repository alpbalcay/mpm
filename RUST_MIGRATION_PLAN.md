# Rust Migration Plan: CB-Geo MPM

## Executive Summary

Migrate the CB-Geo Material Point Method codebase (~20K LOC headers/source + 34K LOC tests) from C++14 to idiomatic Rust while preserving all functionality, performance characteristics, and the existing JSON-based input format.

---

## Phase 0: Project Scaffolding & Build Infrastructure

### 0.1 Cargo Workspace Setup
Create a Cargo workspace with multiple crates to mirror the modular C++ architecture:

```
mpm-rs/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── mpm-core/           # Core types: Index, VectorDim, MatrixDim, data_types
│   ├── mpm-elements/       # Element shape functions, quadrature rules
│   ├── mpm-materials/      # Constitutive models
│   ├── mpm-domain/         # Node, Cell, Particle, Mesh
│   ├── mpm-solver/         # MPM solvers, schemes (USF/USL)
│   ├── mpm-io/             # I/O: JSON config, mesh reading, VTK/HDF5 output
│   ├── mpm-contact/        # Contact mechanics
│   ├── mpm-bcs/            # Boundary conditions, loads, constraints
│   └── mpm-functions/      # Time-dependent loading functions
├── src/
│   └── main.rs             # CLI entry point
├── tests/                  # Integration tests
└── benches/                # Benchmarks
```

### 0.2 Dependency Selection (Rust equivalents for C++ libraries)

| C++ Library | Rust Replacement | Crate |
|---|---|---|
| Eigen3 | `nalgebra` | Dense linear algebra, fixed-size vectors/matrices |
| nlohmann/json | `serde` + `serde_json` | JSON parsing with derive macros |
| spdlog | `tracing` + `tracing-subscriber` | Structured logging |
| tclap | `clap` (derive) | CLI argument parsing |
| Catch2 | Built-in `#[cfg(test)]` + `assert!` macros | Unit/integration testing |
| tsl::robin_map | `hashbrown::HashMap` (default in std) | Fast hash maps |
| Boost::filesystem | `std::fs` / `std::path` | Filesystem operations |
| Boost::uuid | `uuid` crate | UUID generation |
| HDF5 | `hdf5` crate | HDF5 file I/O |
| MPI | `mpi` crate (rsmpi) | MPI bindings |
| OpenMP | `rayon` | Data-parallel iteration |
| VTK output | `vtkio` crate | VTK file writing |
| KaHIP | `metis` crate or FFI to KaHIP | Graph partitioning |

### 0.3 Feature Flags (replacing `#ifdef` conditional compilation)

```toml
[features]
default = ["hdf5-output"]
mpi = ["dep:mpi"]
hdf5-output = ["dep:hdf5"]
vtk-output = ["dep:vtkio"]
partio-output = []
mkl = []  # Link Intel MKL for BLAS/LAPACK
graph-partitioning = ["mpi", "dep:metis"]
```

---

## Phase 1: Core Types & Linear Algebra Foundation (`mpm-core`)

**C++ source:** `include/data_types.h`, `include/mutex.h`

### 1.1 Type Aliases with `nalgebra`
```rust
use nalgebra as na;

pub type Index = u64;
pub type VectorDim<const D: usize> = na::SVector<f64, D>;
pub type MatrixDim<const D: usize> = na::SMatrix<f64, D, D>;
pub type Vector6d = na::SVector<f64, 6>;  // Voigt notation
pub type Matrix6d = na::SMatrix<f64, 6, 6>;
```

### 1.2 Dimension Generics Strategy

The C++ code templates on `unsigned Tdim`. In Rust, use **const generics**:

```rust
pub struct Node<const DIM: usize> { ... }
pub struct Cell<const DIM: usize> { ... }
pub struct Particle<const DIM: usize> { ... }
pub struct Mesh<const DIM: usize> { ... }
```

Instantiate at call sites with `Mesh<2>` or `Mesh<3>`.

### 1.3 Factory Pattern Replacement

C++ uses a singleton `Factory<Tbase, ...Targs>` with string-keyed registration. In Rust, replace with:

- **Enum dispatch** for closed sets (element types, material models, solver types)
- **`inventory` crate** or manual registry for open plugin-style registration
- Deserialization-driven construction via `serde` tagged enums

```rust
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum MaterialConfig<const DIM: usize> {
    LinearElastic { density: f64, youngs_modulus: f64, poisson_ratio: f64 },
    MohrCoulomb { density: f64, youngs_modulus: f64, ... },
    ModifiedCamClay { ... },
    NorSand { ... },
    Newtonian { ... },
    Bingham { ... },
}
```

This replaces both the factory and JSON parsing in one step.

**Estimated effort:** ~500 LOC

---

## Phase 2: Elements & Quadrature (`mpm-elements`)

**C++ source:** `include/elements/` — 4,570 LOC across 17 files

### 2.1 Element Trait

```rust
pub trait Element<const DIM: usize>: Send + Sync {
    fn nfunctions(&self) -> usize;
    fn shapefn(&self, xi: &VectorDim<DIM>, particle_size: &VectorDim<DIM>,
               deformation_gradient: &MatrixDim<DIM>) -> na::DVector<f64>;
    fn grad_shapefn(&self, xi: &VectorDim<DIM>, particle_size: &VectorDim<DIM>,
                    deformation_gradient: &MatrixDim<DIM>) -> na::DMatrix<f64>;
    fn local_coordinates(&self, point: &VectorDim<DIM>,
                         nodal_coords: &na::DMatrix<f64>) -> Option<VectorDim<DIM>>;
    // ... other methods from Element<Tdim> base class
}
```

### 2.2 Element Implementations (port order)

1. **QuadrilateralElement** (4/8/9-node) — 2D reference, 1,085 LOC
2. **TriangleElement** (3/6-node) — 688 LOC
3. **HexahedronElement** (8/20/27-node) — 3D reference, 959 LOC
4. **QuadrilateralGIMPElement** — 418 LOC
5. **HexahedronGIMPElement** — 369 LOC

### 2.3 Quadrature Rules

Port quadrature point/weight tables as static arrays:

```rust
pub trait Quadrature<const DIM: usize> {
    fn points(&self) -> &[(VectorDim<DIM>, f64)]; // (point, weight) pairs
}
```

Implementations: `QuadrilateralQuadrature`, `TriangleQuadrature`, `HexahedronQuadrature`

**Estimated effort:** ~3,000 LOC (mostly numerical constants and shape function math)

---

## Phase 3: Material Models (`mpm-materials`)

**C++ source:** `include/materials/` — 2,133 LOC across 16 files

### 3.1 Material Trait

```rust
pub trait Material<const DIM: usize>: Send + Sync {
    fn id(&self) -> Index;
    fn density(&self) -> f64;
    fn compute_stress(
        &self,
        stress: &Vector6d,
        dstrain: &Vector6d,
        particle: &dyn ParticleData,
        state_vars: &mut StateVariables,
    ) -> Vector6d;
    fn thermodynamic_pressure(&self, ...) -> f64;
    fn initialise_state_variables(&self) -> StateVariables;
}
```

### 3.2 Material Utility Functions

Port `material_utility.h/tcc` (308 LOC):
- `principal_stress()` — eigenvalue decomposition
- `dp_dsigma()` — stress derivatives
- `dq_dsigma()` — deviatoric stress derivatives
- `dtheta_dsigma()` — Lode angle derivatives
- Voigt notation conversions

### 3.3 Material Implementations (port order by complexity)

1. **LinearElastic** — 155 LOC, simplest, good validation target
2. **Newtonian** — 196 LOC, fluid model
3. **Bingham** — 223 LOC, viscoplastic fluid
4. **MohrCoulomb** — 602 LOC, Mohr-Coulomb plasticity with tension cutoff
5. **ModifiedCamClay** — 848 LOC, critical state soil model
6. **NorSand** — 710 LOC, advanced sand model (most complex)

### 3.4 State Variables

Replace C++ `std::map<std::string, double>` with a typed struct per material:

```rust
pub enum StateVariables {
    LinearElastic,
    MohrCoulomb(MohrCoulombState),
    ModifiedCamClay(ModifiedCamClayState),
    NorSand(NorSandState),
    // ...
}
```

**Estimated effort:** ~2,000 LOC

---

## Phase 4: Domain Objects (`mpm-domain`)

**C++ source:** `include/node.h`, `node_base.h`, `cell.h`, `particles/`, `mesh.h`, `nodal_properties.h`, `containers/` — ~8,600 LOC

### 4.1 Node

```rust
pub struct Node<const DIM: usize> {
    id: Index,
    coordinates: VectorDim<DIM>,
    dof: [bool; DIM],            // degree of freedom activity
    mass: [f64; N_PHASES],
    velocity: [VectorDim<DIM>; N_PHASES],
    momentum: [VectorDim<DIM>; N_PHASES],
    internal_force: [VectorDim<DIM>; N_PHASES],
    external_force: [VectorDim<DIM>; N_PHASES],
    // ... additional fields
}
```

Port: `node_base.h` (259 LOC) + `node.h` (327 LOC) + `node.tcc` (633 LOC) = ~1,219 LOC

### 4.2 Cell

```rust
pub struct Cell<const DIM: usize> {
    id: Index,
    nnodes: usize,
    nodes: Vec<Arc<RwLock<Node<DIM>>>>,
    element: Arc<dyn Element<DIM>>,
    particle_ids: Vec<Index>,
    volume: f64,
    centroid: VectorDim<DIM>,
    // ...
}
```

Port: `cell.h` (274 LOC) + `cell.tcc` (846 LOC) = ~1,120 LOC

### 4.3 Particle

```rust
pub struct Particle<const DIM: usize> {
    id: Index,
    coordinates: VectorDim<DIM>,
    cell_id: Option<Index>,
    mass: f64,
    volume: f64,
    density: f64,
    velocity: VectorDim<DIM>,
    stress: Vector6d,
    strain: Vector6d,
    material: Arc<dyn Material<DIM>>,
    state_variables: StateVariables,
    // ... many more fields
}
```

Port: `particle_base.h` (346 LOC) + `particle.h` (414 LOC) + `particle.tcc` (1,162 LOC) = ~1,922 LOC

### 4.4 Mesh (largest single module)

```rust
pub struct Mesh<const DIM: usize> {
    nodes: HashMap<Index, Arc<RwLock<Node<DIM>>>>,
    cells: HashMap<Index, Arc<RwLock<Cell<DIM>>>>,
    particles: HashMap<Index, Arc<RwLock<Particle<DIM>>>>,
    materials: HashMap<Index, Arc<dyn Material<DIM>>>,
    // ... boundary conditions, contacts, etc.
}
```

Port: `mesh.h` (540 LOC) + `mesh.tcc` (2,018 LOC) = ~2,558 LOC

### 4.5 Containers

The C++ `Vector<T>` and `Map<T>` wrappers add thread-safe ID-based lookup. In Rust, replace with:
- `HashMap<Index, Arc<RwLock<T>>>` for concurrent access
- Or `DashMap<Index, T>` (concurrent hash map crate)

### 4.6 Concurrency Model Decision

C++ uses `shared_ptr` + custom `SpinMutex`. Rust options:

- **Option A (recommended):** `Arc<RwLock<T>>` for shared mutable domain objects. Rayon for parallel iteration over particle/node collections.
- **Option B:** Arena allocation with index-based references (ECS-like). Better cache performance but larger refactor.
- **Option C:** Split data into SoA (struct-of-arrays) for hot loops, AoS for management.

Recommend **Option A** initially for correctness, with Option C as a performance optimization in a later phase.

**Estimated effort:** ~5,000 LOC

---

## Phase 5: Boundary Conditions, Loads & Functions (`mpm-bcs`, `mpm-functions`)

**C++ source:** `include/loads_bcs/` (302 LOC), `include/functions/` (121 LOC), `include/contacts/` (120 LOC)

### 5.1 Constraints & Loads

```rust
pub struct VelocityConstraint {
    pub node_id: Index,
    pub dir: usize,
    pub velocity: f64,
}

pub struct FrictionConstraint {
    pub node_id: Index,
    pub dir: usize,
    pub sign_n: i32,
    pub friction: f64,
}

pub struct Traction {
    pub particle_id: Index,
    pub dir: usize,
    pub traction: f64,
}
```

### 5.2 Time-Dependent Functions

```rust
pub trait Function: Send + Sync {
    fn value(&self, t: f64) -> f64;
}
pub struct LinearFunction { ... }
pub struct SinFunction { ... }
```

### 5.3 Contact

```rust
pub trait Contact<const DIM: usize>: Send + Sync {
    fn compute_contact_forces(&self, mesh: &mut Mesh<DIM>);
}
pub struct ContactFriction<const DIM: usize> { ... }
```

**Estimated effort:** ~500 LOC

---

## Phase 6: I/O (`mpm-io`)

**C++ source:** `include/io/` (1,107 LOC) + `src/io/` (~568 LOC) + `include/hdf5_particle.h` (72 LOC)

### 6.1 Configuration Parsing

Replace `IO` class + TCLAP + nlohmann/json with:

```rust
#[derive(Parser)]
#[command(name = "mpm")]
pub struct Cli {
    #[arg(short = 'f', long)]
    pub working_dir: PathBuf,
    #[arg(short = 'i', long, default_value = "mpm.json")]
    pub input_file: String,
    #[arg(short = 'p', long)]
    pub parallel: Option<usize>,
}

#[derive(Deserialize)]
pub struct MpmConfig {
    pub title: String,
    pub mesh: MeshConfig,
    pub particles: Vec<ParticleGeneratorConfig>,
    pub materials: Vec<MaterialConfig>,
    pub analysis: AnalysisConfig,
    // ...
}
```

This replaces ~330 LOC of C++ IO class + JSON parsing.

### 6.2 Mesh I/O (ASCII format)

Port `io_mesh_ascii.h/tcc` (634 LOC):
- Read node coordinates from file
- Read cell connectivity from file
- Read particle coordinates
- Read velocity/friction constraints
- Read tractions

### 6.3 Output Writers

| Writer | C++ LOC | Rust approach |
|--------|---------|--------------|
| VTK | 283 | `vtkio` crate |
| HDF5 | 194 | `hdf5` crate |
| Partio | 48 | Custom writer or FFI |

### 6.4 Logging

Replace spdlog with `tracing`:

```rust
tracing::info!("MPM Analysis");
tracing::debug!("Mesh has {} nodes", mesh.nnodes());
```

**Estimated effort:** ~1,500 LOC

---

## Phase 7: Solvers (`mpm-solver`)

**C++ source:** `include/solvers/` — 1,818 LOC across 11 files

### 7.1 Solver Trait

```rust
pub trait Solver: Send {
    fn solve(&mut self) -> Result<(), MpmError>;
}
```

### 7.2 MPM Base Implementation

Port `mpm_base.h/tcc` (1,446 LOC) — the largest single implementation:
- JSON config reading → `serde` deserialization (much simpler)
- Mesh initialization
- Particle generation & assignment
- Time step loop
- Output scheduling
- Checkpoint/restart

### 7.3 Explicit Solver

Port `mpm_explicit.h/tcc` (287 LOC):

```rust
pub struct MpmExplicit<const DIM: usize> {
    mesh: Mesh<DIM>,
    dt: f64,
    nsteps: u64,
    scheme: Box<dyn MpmScheme<DIM>>,
    // ...
}
```

### 7.4 MPM Schemes

```rust
pub trait MpmScheme<const DIM: usize> {
    fn compute_stress_strain(&self, mesh: &mut Mesh<DIM>, dt: f64, phase: usize);
    fn postcompute_stress_strain(&self, mesh: &mut Mesh<DIM>, dt: f64, phase: usize);
}

pub struct USF;  // Update Stress First
pub struct USL;  // Update Stress Last
```

USF/USL differ in when stress update occurs relative to position update.

### 7.5 Parallelization with Rayon

Replace OpenMP pragmas with Rayon parallel iterators:

```rust
// C++: #pragma omp parallel for
// Rust:
mesh.particles.par_iter_mut().for_each(|particle| {
    particle.compute_strain(dt);
});
```

**Estimated effort:** ~2,000 LOC

---

## Phase 8: MPI Support (Optional Feature)

**C++ source:** `include/graph.h/tcc` (288 LOC), MPI sections throughout mesh/particle/solver

### 8.1 Graph Partitioning

Port `graph.h/tcc` using `metis` crate or KaHIP FFI bindings.

### 8.2 MPI Communication

Use `rsmpi` crate:
- Particle serialization for transfer (use `serde` + `bincode`)
- Halo exchange for ghost cells/nodes
- Domain decomposition

### 8.3 Feature-Gated Code

All MPI code behind `#[cfg(feature = "mpi")]` (replaces `#ifdef USE_MPI`).

**Estimated effort:** ~1,000 LOC

---

## Phase 9: Testing

**C++ source:** 34,193 LOC across 47 test files, 75 TEST_CASEs

### 9.1 Testing Strategy

| C++ Pattern | Rust Equivalent |
|---|---|
| `TEST_CASE("name", "[tag]")` | `#[test] fn name()` |
| `SECTION("section")` | Separate `#[test]` functions or nested blocks |
| `REQUIRE(x == Approx(y).epsilon(e))` | `assert!((x - y).abs() < e)` or `approx` crate |
| Catch2 tags `[2D][3D]` | Separate test modules `mod dim2 { ... }` `mod dim3 { ... }` |

### 9.2 Test Porting Order (matches implementation phases)

1. **Element tests** (9,155 LOC) — validate shape functions against known values
2. **Material tests** (6,585 LOC) — validate stress computations
3. **Node tests** (2,150 LOC) — momentum, force accumulation
4. **Cell tests** (1,877 LOC) — particle assignment, interpolation
5. **Particle tests** (3,029 LOC) — full particle lifecycle
6. **Mesh tests** (4,610 LOC) — mesh construction, neighbor finding
7. **Solver tests** (892 LOC) — end-to-end time stepping
8. **I/O tests** (2,121 LOC) — serialization round-trips
9. **Other tests** (2,774 LOC) — contact, graph, geometry, factory

### 9.3 Test Data

Copy existing test mesh/JSON files from `tests/` to Rust test fixtures.

### 9.4 Validation

Run identical input files through both C++ and Rust versions, compare output particle positions/stresses to machine precision.

**Estimated effort:** ~15,000 LOC (tests are verbose but mechanical to port)

---

## Phase 10: CLI Entry Point & Integration

### 10.1 Main Binary

```rust
fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    tracing_subscriber::init();

    let config: MpmConfig = serde_json::from_reader(
        File::open(cli.working_dir.join(&cli.input_file))?
    )?;

    if let Some(n) = cli.parallel {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }

    let mut solver = create_solver(config)?;
    solver.solve()?;
    Ok(())
}
```

### 10.2 Error Handling

Replace C++ exceptions with `thiserror` / `anyhow`:

```rust
#[derive(thiserror::Error, Debug)]
pub enum MpmError {
    #[error("Invalid material properties: {0}")]
    InvalidMaterial(String),
    #[error("Particle {0} not in any cell")]
    ParticleOutOfDomain(Index),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    // ...
}
```

**Estimated effort:** ~200 LOC

---

## Implementation Order & Dependencies

```
Phase 0: Scaffolding ──────────────────────┐
Phase 1: Core types (nalgebra) ────────────┤
Phase 2: Elements & quadrature ────────────┤
Phase 3: Materials ────────────────────────┤
Phase 5: BCs, loads, functions ────────────┤
                                           ▼
Phase 4: Domain objects (Node, Cell, ──────┐
         Particle, Mesh)                   │
Phase 6: I/O ──────────────────────────────┤
                                           ▼
Phase 7: Solvers ──────────────────────────┐
Phase 8: MPI (optional) ──────────────────┤
                                           ▼
Phase 9: Testing (continuous, per-phase) ──┤
Phase 10: CLI integration ─────────────────┘
```

Phases 1–3 and 5 are independent and can be developed in parallel.

---

## Estimated Total Effort

| Phase | Description | Est. Rust LOC | Complexity |
|-------|-------------|---------------|------------|
| 0 | Scaffolding | 200 | Low |
| 1 | Core types | 500 | Low |
| 2 | Elements | 3,000 | Medium (math-heavy) |
| 3 | Materials | 2,000 | High (numerical algorithms) |
| 4 | Domain objects | 5,000 | High (largest module) |
| 5 | BCs/loads/functions | 500 | Low |
| 6 | I/O | 1,500 | Medium |
| 7 | Solvers | 2,000 | High (core algorithm) |
| 8 | MPI | 1,000 | High (distributed systems) |
| 9 | Tests | 15,000 | Medium (mechanical) |
| 10 | CLI & integration | 200 | Low |
| **Total** | | **~31,000** | |

---

## Key Rust Idiom Translations

| C++ Pattern | Rust Equivalent |
|---|---|
| `template <unsigned Tdim>` | `<const DIM: usize>` const generics |
| `std::shared_ptr<T>` | `Arc<T>` or `Arc<RwLock<T>>` |
| `virtual` methods | `dyn Trait` or enum dispatch |
| `.h` + `.tcc` files | Single `.rs` files per module |
| `#ifdef USE_MPI` | `#[cfg(feature = "mpi")]` |
| `#pragma omp parallel for` | `rayon::par_iter()` |
| `try/catch` exceptions | `Result<T, E>` + `?` operator |
| `Factory<Base>::create("name")` | `serde` tagged enum deserialization |
| `nlohmann::json` | `serde_json` with `#[derive(Deserialize)]` |
| `Eigen::Matrix` | `nalgebra::SMatrix` / `SVector` |
| Catch2 `TEST_CASE` | `#[test]` functions |
| Doxygen `///` | Rustdoc `///` (same convention) |
| `namespace mpm` | `pub mod mpm` (or crate-level) |
| Header guards | Not needed (Rust module system) |

---

## Risk Mitigation

1. **Numerical precision:** Port element/material tests first. Compare output to C++ to ensure floating-point equivalence. Use `approx` crate for tolerance comparisons.

2. **Performance regression:** Benchmark critical paths (P2G, G2P, stress update) early. `nalgebra` with fixed-size matrices should match or exceed Eigen performance.

3. **MPI complexity:** Defer MPI to last phase. Ensure single-node correctness first. `rsmpi` is less mature than C++ MPI — may need unsafe FFI fallbacks.

4. **Large module risk (Mesh/MPMBase):** These are 2,500+ LOC each in C++. Break into smaller Rust submodules: `mesh/construction.rs`, `mesh/particles.rs`, `mesh/constraints.rs`, etc.

5. **JSON backward compatibility:** Ensure Rust reads the exact same `mpm.json` input files. Write serde schemas that match existing JSON structure precisely.
