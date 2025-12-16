use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use ndarray::{Array2, ArrayView3};

mod lane_detect;
use lane_detect::detect_lanes;

/// Python wrapper: accept HxWx3 uint8 NumPy array and return Nx4 float64 NumPy array
#[pyfunction]
fn detect_lanes_rust<'py>(py: Python<'py>, frame: PyReadonlyArray3<'py, u8>) -> PyResult<Py<PyArray2<f64>>> {
    let view: ArrayView3<u8> = frame.as_array();
    let lines = detect_lanes(&view).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let num = lines.len();
    let mut data: Vec<f64> = Vec::with_capacity(num * 4);
    for (x1, y1, x2, y2) in lines {
        data.push(x1);
        data.push(y1);
        data.push(x2);
        data.push(y2);
    }

    let arr = Array2::from_shape_vec((num as usize, 4), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("array creation failed: {}", e)))?;
    Ok(arr.into_pyarray(py).to_owned())
}

#[pymodule]
fn adas_pilot(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_lanes_rust, m)?)?;
    Ok(())
}