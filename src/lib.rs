use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3};
use ndarray::{Array2, ArrayView3};
use image::{ImageBuffer, Rgb, imageops::FilterType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::inputs;
use ort::value::Tensor;

mod object_proc;
mod lane_detect;
mod lane_manager;
mod traffic_light;

use object_proc::ObjectTracker;
use traffic_light::{detect_traffic_light, LightStatus};

#[pyfunction]
fn detect_lanes<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'_, u8>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let frame_view: ArrayView3<u8> = frame.as_array();
    
    // Call our "Pure Rust" lane detection
    let lines = lane_detect::detect_lanes(&frame_view).map_err(|e| {
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

    let py_array = arr.into_pyarray_bound(py);
    
    Ok(py_array)
}

#[pyclass]
struct Tracker {
    inner: ObjectTracker,
}

#[pymethods]
impl Tracker {
    #[new]
    fn new() -> Self {
        Tracker {
            inner: ObjectTracker::new()
        }
    }

    fn process_frame(
        &mut self, 
        detections: Vec<(f64, f64, f64, f64)>, 
        dt: f64
    ) -> Vec<(usize, f64, f64, f64, f64, f64, f64, f64)> {
        let results = self.inner.process_frame(detections, dt);
        results.into_iter().map(|o| {
            (o.id, o.bbox.0, o.bbox.1, o.bbox.2, o.bbox.3, o.distance, o.speed, o.collisiontime)
        }).collect()
    }
}

#[pyclass]
struct LaneManager {
    inner: lane_manager::LaneManager,
}

#[pymethods]
impl LaneManager {
    #[new]
    #[pyo3(signature = (smoothing=0.6, is_two_way=false))]
    fn new(smoothing: f64, is_two_way:bool) -> Self {
        LaneManager {
            inner: lane_manager::LaneManager::new(smoothing, is_two_way)
        }
    }

    fn update_lanes(&mut self, raw_lines: Vec<(f64, f64, f64, f64)>, img_width:f64) -> ((f64, f64, f64, f64), (f64, f64, f64, f64)) {
        let (l, r) = self.inner.update_lines(raw_lines, img_width);
        (l.unwrap_or((0.0,0.0,0.0,0.0)), r.unwrap_or((0.0,0.0,0.0,0.0)))
    }

    fn filter_objects(&self, detections:Vec<(f64, f64, f64, f64)>) -> Vec<((f64, f64, f64, f64), bool)> {
        let results = self.inner.filter_objects(detections);
        results.into_iter().map(|obj| (obj.bbox, obj.is_in_ego_lane)).collect()
    }
}



#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4],
    conf: f32,
    class_id: usize,
}

fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    intersection / (area1 + area2 - intersection)
}

#[pyclass]
pub struct AdasBrain {
    session: Session,
    light_session: Session,
}

#[pymethods]
impl AdasBrain {
    #[new]
        pub fn new(model_path: &str, light_model_path: &str) -> PyResult<Self> {
        
        // 1. Traffic Sign Engine
            let session = Session::builder()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_execution_providers([ort::CUDAExecutionProvider::default().build()]))
                .and_then(|b| b.commit_from_file(model_path))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load Traffic Sign ONNX: {}", e)))?;

        // 2. Traffic Light Engine
            let light_session = Session::builder()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_execution_providers([ort::CUDAExecutionProvider::default().build()]))
                .and_then(|b| b.commit_from_file(light_model_path))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load Traffic Light ONNX: {}", e)))?;

        Ok(AdasBrain { session, light_session })
    }

    pub fn process_frame(
        &mut self,
        py: Python,
        frame_bytes: &[u8],
        orig_width: u32,
        orig_height: u32,
        conf_threshold: f32,
    ) -> PyResult<PyObject> {
        
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(orig_width, orig_height, frame_bytes.to_vec())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid image dimensions or bytes"))?;

        let resized = image::imageops::resize(&img, 640, 640, FilterType::Triangle);
        let mut input_data = vec![0.0f32; 3 * 640 * 640];
        
        for (x, y, pixel) in resized.enumerate_pixels() {
            let r_idx = 0 * (640 * 640) + (y as usize * 640) + x as usize;
            let g_idx = 1 * (640 * 640) + (y as usize * 640) + x as usize;
            let b_idx = 2 * (640 * 640) + (y as usize * 640) + x as usize;

            input_data[r_idx] = (pixel[0] as f32) / 255.0; 
            input_data[g_idx] = (pixel[1] as f32) / 255.0; 
            input_data[b_idx] = (pixel[2] as f32) / 255.0; 
        }

        let shape = [1usize, 3, 640, 640]; 
        let input_tensor = Tensor::from_array((shape, input_data))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let model_inputs = inputs!["images" => input_tensor];
        let outputs = self.session.run(model_inputs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let (_output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut raw_detections = Vec::new();
        let num_guesses = 8400;
        let num_classes = 85; 

        for i in 0..num_guesses {
            let mut max_conf = 0.0;
            let mut class_id = 0;

            for c in 0..num_classes {
                let conf = output_data[(4 + c) * num_guesses + i];
                if conf > max_conf { max_conf = conf; class_id = c; }
            }

            if max_conf > conf_threshold {
                let xc = output_data[0 * num_guesses + i];
                let yc = output_data[1 * num_guesses + i];
                let w  = output_data[2 * num_guesses + i];
                let h  = output_data[3 * num_guesses + i];

                let x1 = xc - w / 2.0;
                let y1 = yc - h / 2.0;
                let x2 = xc + w / 2.0;
                let y2 = yc + h / 2.0;

                let scale_x = orig_width as f32 / 640.0;
                let scale_y = orig_height as f32 / 640.0;

                raw_detections.push(Detection {
                    bbox: [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    conf: max_conf,
                    class_id,
                });
            }
        }

        raw_detections.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap());
        let mut final_detections: Vec<Detection> = Vec::new();

        for detection in raw_detections {
            let mut keep = true;
            for kept_box in &final_detections {
                if detection.class_id == kept_box.class_id && calculate_iou(&detection.bbox, &kept_box.bbox) > 0.45 {
                    keep = false;
                    break;
                }
            }
            if keep { final_detections.push(detection.clone()); }
        }

        let py_results = PyList::empty_bound(py);
        for det in final_detections {
            let dict = PyDict::new_bound(py);
            dict.set_item("class_id", det.class_id)?;
            dict.set_item("conf", det.conf)?;
            dict.set_item("bbox", vec![det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])?;
            py_results.append(dict)?;
        }

        Ok(py_results.into())
    }
}

// ==========================================
// MODULE REGISTRATION
// ==========================================

#[pymodule]
fn adas_pilot(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_lanes, m)?)?;
    m.add_function(wrap_pyfunction!(check_traffic_lights, m)?)?;
    m.add_class::<Tracker>()?;
    m.add_class::<LaneManager>()?;
    
    m.add_class::<AdasBrain>()?;
    
    Ok(())
}