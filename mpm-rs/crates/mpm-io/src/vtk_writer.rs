//! VTK output writer for particle visualization.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Write particle geometry to VTK format (legacy ASCII).
pub fn write_geometry(
    filename: &Path,
    coordinates: &[[f64; 3]],
) -> io::Result<()> {
    let mut file = fs::File::create(filename)?;

    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "MPM Particle Data")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET POLYDATA")?;
    writeln!(file, "POINTS {} double", coordinates.len())?;

    for coord in coordinates {
        writeln!(file, "{} {} {}", coord[0], coord[1], coord[2])?;
    }

    writeln!(file, "VERTICES {} {}", coordinates.len(), 2 * coordinates.len())?;
    for i in 0..coordinates.len() {
        writeln!(file, "1 {}", i)?;
    }

    Ok(())
}

/// Write scalar point data to VTK format.
pub fn write_scalar_point_data(
    filename: &Path,
    coordinates: &[[f64; 3]],
    data: &[f64],
    data_field: &str,
) -> io::Result<()> {
    let mut file = fs::File::create(filename)?;

    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "MPM Particle Data")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET POLYDATA")?;
    writeln!(file, "POINTS {} double", coordinates.len())?;

    for coord in coordinates {
        writeln!(file, "{} {} {}", coord[0], coord[1], coord[2])?;
    }

    writeln!(file, "VERTICES {} {}", coordinates.len(), 2 * coordinates.len())?;
    for i in 0..coordinates.len() {
        writeln!(file, "1 {}", i)?;
    }

    writeln!(file, "POINT_DATA {}", data.len())?;
    writeln!(file, "SCALARS {} double 1", data_field)?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for val in data {
        writeln!(file, "{}", val)?;
    }

    Ok(())
}

/// Write vector point data to VTK format.
pub fn write_vector_point_data(
    filename: &Path,
    coordinates: &[[f64; 3]],
    data: &[[f64; 3]],
    data_field: &str,
) -> io::Result<()> {
    let mut file = fs::File::create(filename)?;

    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "MPM Particle Data")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET POLYDATA")?;
    writeln!(file, "POINTS {} double", coordinates.len())?;

    for coord in coordinates {
        writeln!(file, "{} {} {}", coord[0], coord[1], coord[2])?;
    }

    writeln!(file, "VERTICES {} {}", coordinates.len(), 2 * coordinates.len())?;
    for i in 0..coordinates.len() {
        writeln!(file, "1 {}", i)?;
    }

    writeln!(file, "POINT_DATA {}", data.len())?;
    writeln!(file, "VECTORS {} double", data_field)?;
    for val in data {
        writeln!(file, "{} {} {}", val[0], val[1], val[2])?;
    }

    Ok(())
}

/// Write mesh connectivity to VTK format (for background grid).
pub fn write_mesh(
    filename: &Path,
    coordinates: &[[f64; 3]],
    node_pairs: &[[u64; 2]],
) -> io::Result<()> {
    let mut file = fs::File::create(filename)?;

    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "MPM Mesh Data")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET POLYDATA")?;
    writeln!(file, "POINTS {} double", coordinates.len())?;

    for coord in coordinates {
        writeln!(file, "{} {} {}", coord[0], coord[1], coord[2])?;
    }

    writeln!(file, "LINES {} {}", node_pairs.len(), 3 * node_pairs.len())?;
    for pair in node_pairs {
        writeln!(file, "2 {} {}", pair[0], pair[1])?;
    }

    Ok(())
}
