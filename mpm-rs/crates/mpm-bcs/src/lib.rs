//! Boundary conditions and loads for the MPM library.

use mpm_functions::Function;
use std::sync::Arc;

/// Velocity constraint on a node.
#[derive(Debug, Clone)]
pub struct VelocityConstraint {
    pub setid: i32,
    pub dir: usize,
    pub velocity: f64,
}

impl VelocityConstraint {
    pub fn new(setid: i32, dir: usize, velocity: f64) -> Self {
        Self {
            setid,
            dir,
            velocity,
        }
    }
}

/// Friction constraint on a node.
#[derive(Debug, Clone)]
pub struct FrictionConstraint {
    pub setid: i32,
    pub dir: usize,
    pub sign_n: i32,
    pub friction: f64,
}

impl FrictionConstraint {
    pub fn new(setid: i32, dir: usize, sign_n: i32, friction: f64) -> Self {
        Self {
            setid,
            dir,
            sign_n,
            friction,
        }
    }
}

/// Traction load on particles.
#[derive(Clone)]
pub struct Traction {
    pub setid: i32,
    pub traction_fn: Option<Arc<dyn Function>>,
    pub dir: usize,
    pub traction: f64,
}

impl Traction {
    pub fn new(
        setid: i32,
        traction_fn: Option<Arc<dyn Function>>,
        dir: usize,
        traction: f64,
    ) -> Self {
        Self {
            setid,
            traction_fn,
            dir,
            traction,
        }
    }

    /// Return the traction value at the current time.
    pub fn value(&self, current_time: f64) -> f64 {
        match &self.traction_fn {
            Some(func) => self.traction * func.value(current_time),
            None => self.traction,
        }
    }
}

/// Particle injection configuration.
#[derive(Debug, Clone)]
pub struct Injection {
    pub cell_set_id: i32,
    pub particle_type: String,
    pub material_ids: Vec<u32>,
    pub nparticles_per_dir: usize,
    pub duration: (f64, f64),
    pub velocity: Option<[f64; 3]>,
}
