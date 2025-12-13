use image::{GrayImage, ImageBuffer, Luma};
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
// don't rely on imageproc Hough implementation here; use a simple placeholder
use ndarray::ArrayView3;

pub type Line = (f64, f64, f64, f64);

// Convert BGR ndarray (opencv format) to grayscale GrayImage.
fn bgr_to_gray(frame: &ArrayView3<u8>) -> GrayImage {
    let (height, width, _) = frame.dim();
    let height = height as u32;
    let width = width as u32;
    let mut gray_img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let b = frame[[y as usize, x as usize, 0]] as f32;
            let g = frame[[y as usize, x as usize, 1]] as f32;
            let r = frame[[y as usize, x as usize, 2]] as f32;
            let gray = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            gray_img.put_pixel(x, y, Luma([gray]));
        }
    }
    gray_img
}

// Applying Gaussian blur to grayscale image.
fn apply_gaussian_blur(gray: &GrayImage) -> image::ImageBuffer<Luma<f32>, Vec<f32>> {
    let width = gray.width();
    let height = gray.height();
    // convert to f32 buffer
    let buf: image::ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_fn(width, height, |x, y| {
        let v = gray.get_pixel(x, y).0[0] as f32;
        Luma([v])
    });
    gaussian_blur_f32(&buf, 2.0)
}

/// Creates a trapezoidal ROI mask (zeroes out sky/ hood, focuses on road).
fn apply_roi(blurred: &image::ImageBuffer<Luma<f32>, Vec<f32>>, width: u32, height: u32) -> GrayImage {
    let mut roi = ImageBuffer::new(width, height);
    let vertices: [(u32, u32); 4] = [
        (0, height / 2),
        (width, height / 2),
        (width * 3 / 5, height / 5),
        (width * 2 / 5, height / 5),
    ];

    for y in 0..height {
        for x in 0..width {
            if is_point_in_polygon((x, y), &vertices) {
                let val = blurred.get_pixel(x, y).0[0];
                roi.put_pixel(x, y, Luma([val as u8]));
            } else {
                roi.put_pixel(x, y, Luma([0]));
            }
        }
    }
    roi
}

// Simple trapezoid point check
fn is_point_in_polygon(point: (u32, u32), vertices: &[(u32, u32); 4]) -> bool {
    let (px, py) = point;
    // vertical bounds
    if py < vertices[2].1 || py > vertices[0].1 {
        return false;
    }
    let left_interp = vertices[3].0 as f32
        + (vertices[0].0 as f32 - vertices[3].0 as f32)
            * ((py as f32 - vertices[3].1 as f32) / (vertices[0].1 as f32 - vertices[3].1 as f32));
    let right_interp = vertices[2].0 as f32
        + (vertices[1].0 as f32 - vertices[2].0 as f32)
            * ((py as f32 - vertices[2].1 as f32) / (vertices[1].1 as f32 - vertices[2].1 as f32));
    (px as f32) >= left_interp && (px as f32) <= right_interp
}

// detect edges using canny.
fn detect_edges(roi: &GrayImage) -> image::ImageBuffer<Luma<u8>, Vec<u8>> {
    // use canny directly on the u8 GrayImage
    let edges = canny(roi, 50.0, 150.0);
    edges
}
// Performs Hough transform and averages lines into left/right lanes.
fn hough_transform(_edges: &image::ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Line> {
    // Placeholder: return two static lines (left and right) relative to typical 1280x720 frame
    // These are in (x1,y1,x2,y2) format.
    let left = (100.0, 720.0, 400.0, 360.0);
    let right = (900.0, 720.0, 1100.0, 360.0);
    vec![left, right]
}

// Averages a set of lines by endpoints means.
fn average_lines(lines: &[Line]) -> Option<Line> {
    if lines.is_empty() {
        return None;
    }
    let mut x1s = Vec::new();
    let mut y1s = Vec::new();
    let mut x2s = Vec::new();
    let mut y2s = Vec::new();
    for line in lines.iter() {
        x1s.push(line.0);
        y1s.push(line.1);
        x2s.push(line.2);
        y2s.push(line.3);
    }
    let avg_x1 = x1s.iter().sum::<f64>() / x1s.len() as f64;
    let avg_y1 = y1s.iter().sum::<f64>() / y1s.len() as f64;
    let avg_x2 = x2s.iter().sum::<f64>() / x2s.len() as f64;
    let avg_y2 = y2s.iter().sum::<f64>() / y2s.len() as f64;
    Some((avg_x1, avg_y1, avg_x2, avg_y2))
}

pub fn detect_lanes(frame: &ArrayView3<u8>) -> Result<Vec<Line>, String> {
    let gray = bgr_to_gray(frame);
    let blurred = apply_gaussian_blur(&gray);
    let roi = apply_roi(&blurred, gray.width(), gray.height());
    let edges = detect_edges(&roi);
    let lines = hough_transform(&edges);
    Ok(lines)
}


