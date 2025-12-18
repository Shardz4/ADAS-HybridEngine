use image::{GrayImage, ImageBuffer, Luma}; // Removed unused 'Pixel'
use image::imageops; // Removed unused 'GenericImageView'
use imageproc::edges::canny;
use imageproc::hough::{detect_lines, LineDetectionOptions, PolarLine};
use ndarray::ArrayView3;
use std::f64::consts::PI;

pub type Line = (f64, f64, f64, f64); // (x1, y1, x2, y2)

/// Converts BGR ndarray to grayscale GrayImage.
fn bgr_to_gray(frame: &ArrayView3<u8>) -> GrayImage {
    let (height, width, _) = frame.dim();
    let mut gray_img = ImageBuffer::new(width as u32, height as u32);
    for (x, y, pixel) in gray_img.enumerate_pixels_mut() {
        let bgr = frame[[y as usize, x as usize, 0]];
        let g = frame[[y as usize, x as usize, 1]];
        let r = frame[[y as usize, x as usize, 2]];
        let gray = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * bgr as f32;
        *pixel = Luma([gray as u8]);
    }
    gray_img
}

/// Applies Gaussian blur.
fn apply_gaussian_blur(gray: &GrayImage) -> GrayImage {
    imageops::blur(gray, 2.0)
}

/// Creates a trapezoidal ROI mask.
fn apply_roi(blurred: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut roi = ImageBuffer::new(width, height);
    let vertices: [(u32, u32); 4] = [
        (0, height),                     // Bottom-left
        (width, height),                 // Bottom-right
        (width * 3 / 5, height * 3 / 5), // Top-right
        (width * 2 / 5, height * 3 / 5), // Top-left
    ];

    for y in 0..height {
        for x in 0..width {
            if is_point_in_polygon((x, y), &vertices) {
                let val = blurred.get_pixel(x, y).0[0];
                roi.put_pixel(x, y, Luma([val]));
            } else {
                roi.put_pixel(x, y, Luma([0]));
            }
        }
    }
    roi
}

fn is_point_in_polygon(point: (u32, u32), vertices: &[(u32, u32); 4]) -> bool {
    let (px, py) = point;
    if py < vertices[2].1 || py > vertices[0].1 { return false; }
    let height_span = (vertices[0].1 - vertices[3].1) as f32;
    if height_span == 0.0 { return false; }

    let progress = (py - vertices[3].1) as f32 / height_span;
    let left_edge = vertices[3].0 as f32 + (vertices[0].0 as f32 - vertices[3].0 as f32) * progress;
    let right_edge = vertices[2].0 as f32 + (vertices[1].0 as f32 - vertices[2].0 as f32) * progress;

    (px as f32) >= left_edge && (px as f32) <= right_edge
}

/// Detects edges using Canny.
fn detect_edges(roi: &GrayImage) -> GrayImage {
    canny(roi, 50.0, 150.0)
}

/// Performs Hough transform with intelligent filtering and clipping.
fn hough_transform(edges: &GrayImage) -> Vec<Line> {
    // 1. Settings
    let options = LineDetectionOptions {
        suppression_radius: 10,
        vote_threshold: 40, // Slightly lower to catch faint lines
    };

    // 2. Run Detection
    let lines: Vec<PolarLine> = detect_lines(edges, options);
    if lines.is_empty() { return vec![]; }

    let mut left_lines: Vec<Line> = Vec::new();
    let mut right_lines: Vec<Line> = Vec::new();
    
    // Define the "Horizon" (Stop lines 60% up the screen)
    let y_bottom = edges.height() as f64;
    let y_top = edges.height() as f64 * 0.6; 

    for polar in lines {
        let rho = polar.r; 
        let theta_deg = polar.angle_in_degrees;
        let theta_rad = (theta_deg as f64) * PI / 180.0;

        // --- FILTERING ---
        // 1. Reject Vertical lines (theta near 0 or 180)
        if theta_rad < 0.1 || theta_rad > (PI - 0.1) { continue; }
        
        // 2. Reject Horizontal lines (theta near 90 deg / 1.57 rad)
        // This removes the "Horizon" line cutting across the screen
        if (theta_rad - PI / 2.0).abs() < 0.3 { continue; }

        // --- CLIPPING (Make lines fit the road) ---
        // Equation: x = (rho - y * sin(theta)) / cos(theta)
        let x_bottom = (rho as f64 - y_bottom * theta_rad.sin()) / theta_rad.cos();
        let x_top = (rho as f64 - y_top * theta_rad.sin()) / theta_rad.cos();

        let line = (x_bottom, y_bottom, x_top, y_top);

        // Group by slope
        if theta_rad < PI / 2.0 {
            right_lines.push(line);
        } else {
            left_lines.push(line);
        }
    }

    // Return the strongest line for each side
    let mut result = Vec::new();
    if let Some(l) = left_lines.first() { result.push(*l); }
    if let Some(r) = right_lines.first() { result.push(*r); }
    
    result
}

pub fn detect_lanes(frame: &ArrayView3<u8>) -> Result<Vec<Line>, String> {
    let gray = bgr_to_gray(frame);
    let blurred = apply_gaussian_blur(&gray);
    let roi = apply_roi(&blurred, gray.width(), gray.height());
    let edges = detect_edges(&roi);
    let lines = hough_transform(&edges);
    Ok(lines)
}