use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Pixel};
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use imageproc::hough::{detect_lines_polar, PolarLine};
use ndarray::{Array3, ArrayView3};

pub type Line(f64, f64, f64, f64);

// Converting BGR ndarray (opencv fornat) to grayscale GrayImage.

fn bgr_to_gray(frame: &ArrayView<u8>) -> GrayImage {
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

// Applying Gaussian blur to grayscale image.

fn apply_gaussian_blur(gray: &GrayImage) -> image::ImageBuffer<Luma<f32>> {
    let blurred_f32: image::ImageBuffer<Luma<f32>, Vec<f32>> = gray.to_Luma8().to_f32();
    gaussian_blur_f32(&blurred_f32, 2.0)
}

/// Creates a trapezoidal ROI mask (zeroes out sky/ hood, focuses on road).

fn apply_roi(blurred: &image::ImageBuffer<Luma<f32>, Vec<f32>>, width: u32, height: u32) -> GrayImage {
    let mut roi = ImageBuffer::new(width, height);
    let vertices: [(u32, u32); 4] = [
        (0, height / 2),
        (width, height / 2),
        (width *3 / 5, height * 1 / 5),
        (width * 2 / 5, height * 1 / 5),
    ];

    // simple polygoin fill (for production, use imageproc::drawing::draw_polygon);
    for y in 0..height {
        for x in 0..width {
            if is_point_in_polygon((x, y), &vertices) {
                    let val = *blurred.get_pixel(x, y).channels()[0];
                    roi.put_pixel(x, y, Luma([val as u8]));
            } else {
                roi.put_pixel(x, y, Luma([0]));
            }
        }
    }
    roi
}

// Simple point-in-polygon test for trapezoid ROI (trapezoid-specific for efficiency).
fn is_point_in_polygon(point: (u32m u32), vertices: &[(u32, u32); 4]) -> bool{
    let (px, py) = point;
    if py < vertices[2].1 || py > vertices[0].1 {
        return false;
    }
    let left_interp = vertices[3].0 as f32 + (vertices[0].0 as f32 - vertices[3].0 as f32) * ((py as f32 - vertices[3].1 as f32) / (vertices[0].1 as f32 - vertices[3].1 as f32));
    let right_interp = vertices[2].0 as f32 + (vertices[1].0 as f32 - vertices[2].0 as f32) * ((py as f32 - vertices[2].1 as f32) / (vertices[2].1 as f32 - vertices[2].1 as f32));
    (px as f32) >= left_interp && (px as f32) <= right_interp
}

// detect edges using canny.

fn detect_edges(roi: &GrayImage) -> GrayImage {
    let edges_f32: image::ImageBuffer<Luma<f32>, Vec<f32>> = roi.to_luma8().to_f32();
    let edges = canny(&edges_f32, 50.0, 150.0);
    image::ImageBuffer::from_raw(edges.width, edges.height(), edges.into_raw()).unwrap()
}
 // Performs Hough transform and averages lines into left/right lanes.
fn hough_transform(edges: &GrayImage) -> Vec<Line> {
    let lines: Vec<PolarLine> = detect_lines_polar(&edges, 1.0, 1.0, 100.0); // rho_step = 1, theta_step = 1 deg, threshold = 100 votes

    if lines.is_empty() {
        return vec![(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]; // dummy fallback
    }

    // group into  left (theta 0-45 deg) and right (theta 135-180 deg) slopes
    let mut left_lines: Vec<Line> = Vec::new();
    let mut right_lines: Vec<Line> = Vec::new();
    let height = edges.height() as f64;

    for polar in lines{
        let theta_rad = polar.theta;
        let rho = polar.rho;
        if theta_rad<0.0 {
            continue;
        }
        
        // converting polar to cartesiona line

        let a = std::f64::consts::PI / 2.0 - theta_rad;
        let x1 = (rho / a.cos()).cos() * a;
        let y1 = (rho / a.cos()).sin() * a;
        let x2 = (rho - height * theta_rad.sin()) / theta_rad.cos();
        let y2 = height;

        let line = if theta_rad < std::f64::consts::FRAC_PI_4{
            //left lane (+ve slope)
            (x1.max(0.0), y1, x2.max(0.0), y2)
        } else if theta_rad > 3.0 * std::f64::consts::FRAC_PI_4{
            //right lane (-ve slope)
            (x2,y2,x1,y1)
        } else {
            continue;
        };

        if let Some((x1, _, x2, _)) = (line.0 as i32, line.2 as i32){
            if x1 < edgesa.width() as i32 / 2 {
                left_lines.push(line);
            } else {
                right_lines.push(line);
            }
        }

    }
    // averaging lines
    let avg_left = average_lines(&left_lines).unwrap_or((100.0, height, 400.0, height / 2.0));
    let avg_right = average_lines(&right_lines).unwrap_or((900.0, height, 1100.0, height / 2.0));

    vec![avg_left, avg_right]
}



