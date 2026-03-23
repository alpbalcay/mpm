# CLAUDE.md - CB-Geo Material Point Method (MPM)

## Project Overview

CB-Geo MPM is a high-performance C++14 Material Point Method implementation for computational geomechanics. MPM is a hybrid Eulerian-Lagrangian numerical method for large-deformation problems in soil mechanics, geotechnical engineering, and multiphase flow.

## Build & Run

```bash
# Build (from repo root)
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ ..
make -jN              # Full build (executable + tests)
make mpm -jN          # Just the executable
make mpmtest -jN      # Just the test binary

# Run
./mpm -f <working_dir> [-i <input_file>] [-p <threads>]

# MPI
mpirun -N 4 ./mpm -f <working_dir> -i mpm.json
```

### CMake Options

- `-DMPM_BUILD_TESTING=Off` — Disable tests
- `-DHALO_EXCHANGE=On` — Enable MPI halo exchange
- `-DKAHIP_ROOT=<path>` — Domain decomposition (KaHIP)
- `-DPARTIO_ROOT=<path>` — Houdini Partio output

### Required Dependencies

Boost (filesystem, system), Eigen3, HDF5

### Optional Dependencies

MPI, OpenMP, MKL, VTK, Partio, KaHIP

## Testing

```bash
cd build
./mpmtest -s            # Run all tests (verbose)
ctest -VV               # Via CMake test runner
mpirun -N 4 ./mpmtest [mpi]   # MPI-specific tests
```

Tests use Catch2 (bundled in `external/catch.hpp`). Test-to-source ratio is ~3:1 with 34K LOC of tests.

## Linting & Formatting

```bash
# clang-format (80-char column, 2-space indent, CUED GeoMechanics style)
python3 ./clang-tools/run-clang-format.py -r include/* src/* tests/*

# cppcheck
cppcheck --enable=warning --inconclusive --force --language=c++11 src/*.cc include/*.h
```

Configuration lives in `.clang-format`. All code must pass clang-format checks.

## Architecture

### Directory Layout

```
include/           # Headers + template implementations (.h/.tcc) — core library
  elements/        # Finite elements (hex, quad, triangle, GIMP variants)
    2d/            # 2D elements and quadratures
    3d/            # 3D elements and quadratures
  materials/       # Constitutive models (linear elastic, Mohr-Coulomb, etc.)
  solvers/         # MPM solver hierarchy (explicit time integration)
    mpm_scheme/    # USF/USL time stepping schemes
  loads_bcs/       # Boundary conditions and loads
  contacts/        # Contact mechanics
  functions/       # Time-dependent loading functions
  containers/      # Custom vector/map wrappers
  io/              # I/O, mesh reading, VTK/Partio output, logging
src/               # Implementation files (main.cc entry point)
tests/             # Catch2 unit tests
external/          # Bundled third-party: Catch2, nlohmann/json, spdlog, tsl, tclap
cmake/             # CMake find modules
clang-tools/       # Formatting scripts
```

### Key Design Patterns

- **Dimension templating**: Most classes are templated on `Tdim` (2D/3D). Template implementations live in `.tcc` files included at the bottom of `.h` headers.
- **Factory pattern**: `include/factory.h` — singleton registry for creating solvers, materials, and elements by string name from JSON config.
- **Shared pointer ownership**: All domain objects (`Node`, `Cell`, `Particle`, `Material`) managed via `std::shared_ptr`.
- **Namespace**: All code is in `mpm::`.
- **Header guards**: `#ifndef MPM_<MODULE>_H_`

### Class Hierarchies

**Spatial domain:**
- `NodeBase<Tdim>` → `Node<Tdim, Tdof, Tnphases>`
- `ParticleBase<Tdim>` → `Particle<Tdim>`
- `Cell<Tdim>` (contains particles, references element shape functions)
- `Mesh<Tdim>` (top-level container: manages nodes, cells, particles)

**Elements:**
- `Element<Tdim>` → `QuadrilateralElement`, `HexahedronElement`, `TriangleElement`, GIMP variants
- Each has corresponding `Quadrature<Tdim>` classes

**Materials (constitutive models):**
- `Material<Tdim>` → `LinearElastic`, `MohrCoulomb`, `ModifiedCamClay`, `NorSand`, `Newtonian`, `Bingham`
- Stress/strain in Voigt notation (6-component vectors)

**Solvers:**
- `MPM` (interface) → `MPMBase<Tdim>` → `MPMExplicit<Tdim>`
- Schemes: `MPMScheme` → `MPMSchemeUSF` (Update Stress First), `MPMSchemeUSL` (Update Stress Last)

### Core Algorithm (Explicit MPM)

1. Particle-to-Grid (P2G): interpolate particle properties to nodes
2. Compute forces: gravity, boundary conditions, contact
3. Solve momentum: update nodal velocities
4. Grid-to-Particle (G2P): interpolate back to particles
5. Update stress/strain via constitutive model
6. Update particle positions
7. Output: VTK, HDF5 checkpoints

### Parallelization

- **OpenMP**: loop-level parallelism
- **MPI**: domain decomposition with KaHIP graph partitioning, halo exchange, particle migration

### Key Files

| File | Purpose |
|------|---------|
| `src/main.cc` | Entry point: CLI parsing, solver creation, `mpm->solve()` |
| `include/mesh.h` / `.tcc` | Core mesh container (72KB) |
| `include/solvers/mpm_base.h` / `.tcc` | MPM algorithm implementation (43KB) |
| `include/factory.h` | Factory pattern for object creation |
| `include/node.h`, `cell.h`, `particles/particle.h` | Domain objects |
| `include/materials/material.h` | Abstract material interface |
| `include/data_types.h` | Eigen type aliases, MKL configuration |

### Bundled Libraries (in `external/`)

- **tclap**: command-line parsing
- **nlohmann/json**: JSON configuration
- **spdlog**: logging
- **tsl::robin_map**: fast hash map
- **Catch2**: testing framework

## Commit Convention

Commits use emoji prefixes: `:bug:` (bugfix), `:hammer:` (refactor/improvement), `:art:` (style/cleanup), `:heavy_check_mark:` (tests), `:computer:` (feature).

## CI/CD

CircleCI (`.circleci/config.yml`) runs: GCC+MPI build, Clang+static analysis (scan-build), cppcheck + clang-format check, code coverage (lcov), benchmarks. Docker image: `quay.io/cbgeo/mpm`.
