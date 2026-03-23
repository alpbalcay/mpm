//! ASCII mesh I/O: reading node coordinates, cell connectivity, particles.

use mpm_core::{Index, Vector6d, VectorDim};
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

/// Read nodal coordinates from an ASCII mesh file.
///
/// File format: first line is the number of nodes, then each subsequent
/// line contains: node_id coord_x coord_y [coord_z]
pub fn read_mesh_nodes<const DIM: usize>(
    filename: &Path,
) -> io::Result<Vec<VectorDim<DIM>>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut coordinates = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || i == 0 {
            // Skip header line (node count)
            if i == 0 {
                continue;
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < DIM + 1 {
            continue;
        }

        // Skip the node ID (first column)
        let mut coord = VectorDim::<DIM>::zeros();
        for d in 0..DIM {
            coord[d] = parts[d + 1].parse::<f64>().unwrap_or(0.0);
        }
        coordinates.push(coord);
    }
    Ok(coordinates)
}

/// Read cell connectivity from an ASCII mesh file.
///
/// File format: first line is the number of cells, then each line contains:
/// cell_id node1 node2 ... nodeN
pub fn read_mesh_cells(filename: &Path) -> io::Result<Vec<Vec<Index>>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut cells = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || i == 0 {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        // Skip the cell ID (first column)
        let cell_nodes: Vec<Index> = parts[1..]
            .iter()
            .filter_map(|s| s.parse::<Index>().ok())
            .collect();
        cells.push(cell_nodes);
    }
    Ok(cells)
}

/// Read particle coordinates from a file.
pub fn read_particles<const DIM: usize>(
    filename: &Path,
) -> io::Result<Vec<VectorDim<DIM>>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut coordinates = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || i == 0 {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < DIM + 1 {
            continue;
        }

        let mut coord = VectorDim::<DIM>::zeros();
        for d in 0..DIM {
            coord[d] = parts[d + 1].parse::<f64>().unwrap_or(0.0);
        }
        coordinates.push(coord);
    }
    Ok(coordinates)
}

/// Read velocity constraints from a file.
///
/// Format: node_id direction velocity
pub fn read_velocity_constraints(
    filename: &Path,
) -> io::Result<Vec<(Index, usize, f64)>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut constraints = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let node_id = parts[0].parse::<Index>().unwrap_or(0);
            let dir = parts[1].parse::<usize>().unwrap_or(0);
            let velocity = parts[2].parse::<f64>().unwrap_or(0.0);
            constraints.push((node_id, dir, velocity));
        }
    }
    Ok(constraints)
}

/// Read friction constraints from a file.
///
/// Format: node_id direction sign_n friction
pub fn read_friction_constraints(
    filename: &Path,
) -> io::Result<Vec<(Index, usize, i32, f64)>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut constraints = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            let node_id = parts[0].parse::<Index>().unwrap_or(0);
            let dir = parts[1].parse::<usize>().unwrap_or(0);
            let sign_n = parts[2].parse::<i32>().unwrap_or(1);
            let friction = parts[3].parse::<f64>().unwrap_or(0.0);
            constraints.push((node_id, dir, sign_n, friction));
        }
    }
    Ok(constraints)
}

/// Read particle stresses from a file.
///
/// Format: sxx syy szz txy tyz txz (Voigt notation)
pub fn read_particles_stresses(filename: &Path) -> io::Result<Vec<Vector6d>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut stresses = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if parts.len() >= 6 {
            stresses.push(Vector6d::new(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            ));
        }
    }
    Ok(stresses)
}

/// Read particle volumes from a file.
///
/// Format: particle_id volume
pub fn read_particles_volumes(filename: &Path) -> io::Result<Vec<(Index, f64)>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut volumes = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let id = parts[0].parse::<Index>().unwrap_or(0);
            let vol = parts[1].parse::<f64>().unwrap_or(0.0);
            volumes.push((id, vol));
        }
    }
    Ok(volumes)
}

/// Read particle-cell assignments from a file.
///
/// Format: particle_id cell_id
pub fn read_particles_cells(filename: &Path) -> io::Result<Vec<[Index; 2]>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut pairs = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let pid = parts[0].parse::<Index>().unwrap_or(0);
            let cid = parts[1].parse::<Index>().unwrap_or(0);
            pairs.push([pid, cid]);
        }
    }
    Ok(pairs)
}

/// Write particle-cell assignments to a file.
pub fn write_particles_cells(
    filename: &Path,
    pairs: &[[Index; 2]],
) -> io::Result<()> {
    let mut content = String::new();
    for pair in pairs {
        content.push_str(&format!("{}\t{}\n", pair[0], pair[1]));
    }
    fs::write(filename, content)
}

/// Read nodal forces from a file.
///
/// Format: node_id direction force
pub fn read_forces(filename: &Path) -> io::Result<Vec<(Index, usize, f64)>> {
    let file = fs::File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut forces = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let nid = parts[0].parse::<Index>().unwrap_or(0);
            let dir = parts[1].parse::<usize>().unwrap_or(0);
            let force = parts[2].parse::<f64>().unwrap_or(0.0);
            forces.push((nid, dir, force));
        }
    }
    Ok(forces)
}
