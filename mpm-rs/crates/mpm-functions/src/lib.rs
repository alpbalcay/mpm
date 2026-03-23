//! Time-dependent loading functions for the MPM library.

use serde::Deserialize;
use std::sync::Arc;

/// Trait for time-dependent functions.
pub trait Function: Send + Sync {
    /// Evaluate the function at a given input value (typically time).
    fn value(&self, x: f64) -> f64;
}

/// Linear interpolation function defined by tabular data.
#[derive(Debug, Clone)]
pub struct LinearFunction {
    /// Sorted vector of (x, f(x)) pairs for interpolation.
    points: Vec<(f64, f64)>,
}

impl LinearFunction {
    /// Create a new linear function from a JSON configuration.
    ///
    /// Expects `x_values` and `fx_values` arrays in the properties.
    pub fn new(properties: &serde_json::Value) -> Result<Self, String> {
        let x_values = properties
            .get("xvalues")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'xvalues' in linear function properties")?;
        let fx_values = properties
            .get("fxvalues")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'fxvalues' in linear function properties")?;

        if x_values.len() != fx_values.len() {
            return Err("xvalues and fxvalues must have the same length".to_string());
        }

        let mut points: Vec<(f64, f64)> = x_values
            .iter()
            .zip(fx_values.iter())
            .map(|(x, fx)| {
                let x = x.as_f64().unwrap_or(0.0);
                let fx = fx.as_f64().unwrap_or(0.0);
                (x, fx)
            })
            .collect();

        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(Self { points })
    }
}

impl Function for LinearFunction {
    fn value(&self, x: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        // Below first point
        if x <= self.points[0].0 {
            return self.points[0].1;
        }

        // Above last point
        if x >= self.points[self.points.len() - 1].0 {
            return self.points[self.points.len() - 1].1;
        }

        // Linear interpolation between points
        for i in 0..self.points.len() - 1 {
            let (x0, fx0) = self.points[i];
            let (x1, fx1) = self.points[i + 1];
            if x >= x0 && x <= x1 {
                let t = (x - x0) / (x1 - x0);
                return fx0 + t * (fx1 - fx0);
            }
        }

        self.points[self.points.len() - 1].1
    }
}

/// Sinusoidal function: f(x) = sin(a * (x - x0)) within a valid range.
#[derive(Debug, Clone)]
pub struct SinFunction {
    x0: f64,
    a: f64,
    xrange: (f64, f64),
}

impl SinFunction {
    pub fn new(properties: &serde_json::Value) -> Result<Self, String> {
        let x0 = properties
            .get("x0")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let a = properties
            .get("a")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let xrange = match properties.get("xrange").and_then(|v| v.as_array()) {
            Some(arr) if arr.len() >= 2 => {
                let min = arr[0].as_f64().unwrap_or(f64::MIN);
                let max = arr[1].as_f64().unwrap_or(f64::MAX);
                (min, max)
            }
            _ => (f64::MIN, f64::MAX),
        };
        Ok(Self { x0, a, xrange })
    }
}

impl Function for SinFunction {
    fn value(&self, x: f64) -> f64 {
        if x < self.xrange.0 || x > self.xrange.1 {
            return 0.0;
        }
        (self.a * (x - self.x0)).sin()
    }
}

/// Function configuration for deserialization.
#[derive(Debug, Deserialize)]
pub struct FunctionConfig {
    pub id: u64,
    #[serde(rename = "type")]
    pub function_type: String,
    #[serde(flatten)]
    pub properties: serde_json::Value,
}

/// Create a function from configuration.
pub fn create_function(config: &FunctionConfig) -> Result<Arc<dyn Function>, String> {
    match config.function_type.as_str() {
        "Linear" => Ok(Arc::new(LinearFunction::new(&config.properties)?)),
        "Sin" => Ok(Arc::new(SinFunction::new(&config.properties)?)),
        _ => Err(format!("Unknown function type: {}", config.function_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_function() {
        let props = serde_json::json!({
            "xvalues": [0.0, 1.0, 2.0],
            "fxvalues": [0.0, 1.0, 0.5]
        });
        let f = LinearFunction::new(&props).unwrap();
        assert!((f.value(0.0) - 0.0).abs() < 1e-12);
        assert!((f.value(0.5) - 0.5).abs() < 1e-12);
        assert!((f.value(1.0) - 1.0).abs() < 1e-12);
        assert!((f.value(1.5) - 0.75).abs() < 1e-12);
        assert!((f.value(2.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_sin_function() {
        let props = serde_json::json!({
            "x0": 0.0,
            "a": std::f64::consts::PI,
            "xrange": [0.0, 2.0]
        });
        let f = SinFunction::new(&props).unwrap();
        assert!((f.value(0.5) - 1.0).abs() < 1e-12);
        assert!(f.value(1.0).abs() < 1e-12);
        assert!(f.value(3.0).abs() < 1e-12); // out of range
    }
}
