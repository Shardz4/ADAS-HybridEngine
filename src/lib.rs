use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3}; // Fixed: Added missing imports
use ndarray::{Array2, ArrayView3};
use lane_detect::detect_lanes;
use object_proc::ObjectTracker;
mod object_proc;
mod lane_detect;

#[pyfunction]
fn detect_lanes_rust<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray2<f64>>> { // Fixed: Return type updated for PyO3 0.21+
    let frame_view: ArrayView3<u8> = frame.as_array();
    
    // Call our "Pure Rust" lane detection
    let lines = detect_lanes(&frame_view).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Lane detection failed: {}", e))
    })?;

    // Convert Vec<Line> to 2D ndarray (num_lines x 4)
    let num_lines = lines.len();
    let mut data: Vec<f64> = Vec::with_capacity(num_lines * 4);
    for line in lines {
        data.push(line.0); 
        data.push(line.1); 
        data.push(line.2); 
        data.push(line.3);
    }

    let arr = Array2::from_shape_vec((num_lines as usize, 4), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Array creation failed: {}", e)))?;

    // Fixed: .into_pyarray_bound(py) is the new PyO3 0.21 method
    let py_array = arr.into_pyarray_bound(py);
    
    Ok(py_array)
}

#[pyclass]
struct Rusttracker{
    inner: ObjectTracker,
}

#[pymethods]
impl RustTracker{
    #[new]
    fn new() -> Self{
        RustTracker{
            inner: ObjectTracker::new()
        }
    }

    fn process_frame(&mut self, detections: Vec<(f64, f64, f64, f64)>, dt:f64) -> Vec<(uszie, f64, f64, f64, f64, f64, f64, f64)> {
        let results = self.inner.process_frame(detections, dt);
        results.into_iter().map(|o| {
            (o.id, o.bbox.0, o.bbox.1, o.bbox.2, o.bbox.3, o.distance, o.speed, o.collisiontime)
        }).collect()
    }
}

#[pymodule]
fn adas_pilot(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> { // Fixed: Updated signature to remove warning
    m.add_function(wrap_pyfunction!(detect_lanes_rust, m)?)?;
    m.add_class:<RustTracker>()?;
    Ok(())
}