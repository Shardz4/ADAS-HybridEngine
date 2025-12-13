use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray3, ToPyArray};
use ndarray::{Array2, ArrayView3};

mod lane_detect;
use lane_detect::detect_lanes;

#[pymodule]
mod adas_pilot {
    use super::*;
    use pyo3::exceptions::PyValueError;

    /// Detects lane lines in a video frame using Rust-optimized computer vision pipeline.
    #[pyfunction]
    fn detect_lanes_rust<'py>(
        py: Python<'py>,
        frame: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let frame_view: ArrayView3<u8> = frame.as_array();
        let lines = detect_lanes(&frame_view)
            .map_err(|e| PyValueError::new_err(format!("Lane detection failed: {}", e)))?;

        // Flatten Vec<Line> into 1D Vec<f64> for (num_lines, 4) shape
        let num_lines = lines.len() as usize;
        let mut data: Vec<f64> = Vec::with_capacity(num_lines * 4);
        for line in lines {
            data.push(line.0);
            data.push(line.1);
            data.push(line.2);
            data.push(line.3);
        }

        // Construct ndarray and convert to PyArray2
        let shape = (num_lines, 4);
        let arr = Array2::from_shape_vec(shape, data)
            .map_err(|e| PyValueError::new_err(format!("Array creation failed: {}", e)))?;
        let py_array = arr.to_pyarray(py);
        Ok(py_array)
    }
}