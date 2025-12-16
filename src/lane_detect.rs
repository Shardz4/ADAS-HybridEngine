
use ndarray::ArrayView3;
use opencv::{core, imgproc, prelude::*, types};
use std::cell::RefCell;

pub type Line = (f64, f64, f64, f64);

thread_local! {
    // per-thread reusable buffer for Hough output
    static LINES_BUF: RefCell<types::VectorOfVec4i> = RefCell::new(types::VectorOfVec4i::new());
}

/// Detect lanes using OpenCV's Canny + HoughLinesP on a BGR u8 image.
/// Returns Vec<Line> with (x1,y1,x2,y2) in pixel coordinates.
pub fn detect_lanes(frame: &ArrayView3<u8>) -> Result<Vec<Line>, String> {
    let shape = frame.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err("expected HxWx3 BGR uint8 image".into());
    }
    let h = shape[0] as i32;
    let w = shape[1] as i32;

    // require contiguous slice
    let data = frame
        .as_slice()
        .ok_or_else(|| "frame must be C-contiguous".to_string())?;

    // Create Mat from slice and reshape to (h, w, 3)
    let mut mat = Mat::from_slice(data).map_err(|e| e.to_string())?;
    mat = mat.reshape(3, h).map_err(|e| format!("reshape failed: {}", e))?;

    // Convert to gray
    let mut gray = Mat::default();
    imgproc::cvt_color(&mat, &mut gray, imgproc::COLOR_BGR2GRAY, 0).map_err(|e| e.to_string())?;

    // Blur + Canny
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        core::Size::new(5, 5),
        1.5,
        1.5,
        core::BORDER_DEFAULT as i32,
    )
    .map_err(|e| e.to_string())?;

    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false).map_err(|e| e.to_string())?;

    // ROI mask (trapezoid)
    let mut mask = Mat::zeros(h, w, core::CV_8U).map_err(|e| e.to_string())?;
    let pts = types::VectorOfPoint::from_iter(vec![
        core::Point::new((w as f32 * 0.1) as i32, h),
        core::Point::new((w as f32 * 0.4) as i32, (h as f32 * 0.6) as i32),
        core::Point::new((w as f32 * 0.6) as i32, (h as f32 * 0.6) as i32),
        core::Point::new((w as f32 * 0.9) as i32, h),
    ]);
    let mut pts_vec = types::VectorOfVectorOfPoint::new();
    pts_vec.push(pts);
    imgproc::fill_poly(&mut mask, &pts_vec, core::Scalar::all(255.0), imgproc::LINE_8, 0, core::Point::new(0, 0)).map_err(|e| e.to_string())?;

    let mut masked = Mat::default();
    core::bitwise_and(&edges, &mask, &mut masked, &Mat::default()).map_err(|e| e.to_string())?;

    // HoughLinesP with reusable buffer
    LINES_BUF.with(|buf| {
        let mut lines = buf.borrow_mut();
        lines.clear();

        let rho = 1.0;
        let theta = std::f64::consts::PI / 180.0;
        let threshold = 40;
        let min_line_length = 30.0;
        let max_line_gap = 20.0;

        imgproc::hough_lines_p(&masked, &mut *lines, rho, theta, threshold, min_line_length, max_line_gap).map_err(|e| e.to_string())?;

        let mut left_lines: Vec<(f64, f64, f64, f64)> = Vec::new();
        let mut right_lines: Vec<(f64, f64, f64, f64)> = Vec::new();

        for i in 0..lines.len() {
            if let Ok(seg) = lines.get(i) {
                if seg.len() >= 4 {
                    let x1 = seg[0] as f64;
                    let y1 = seg[1] as f64;
                    let x2 = seg[2] as f64;
                    let y2 = seg[3] as f64;
                    if (x2 - x1).abs() < 1e-3 {
                        continue;
                    }
                    let slope = (y2 - y1) / (x2 - x1);
                    if slope.abs() < 0.3 {
                        continue;
                    }
                    if slope < 0.0 {
                        left_lines.push((x1, y1, x2, y2));
                    } else {
                        right_lines.push((x1, y1, x2, y2));
                    }
                }
            }
        }

        let average_line = |segs: &Vec<(f64,f64,f64,f64)>| -> Option<(f64,f64,f64,f64)> {
            if segs.is_empty() { return None; }
            let mut sum_x1=0.0; let mut sum_y1=0.0; let mut sum_x2=0.0; let mut sum_y2=0.0; let mut total_w = 0.0;
            for &(x1,y1,x2,y2) in segs.iter() {
                let w = ((x2-x1).hypot(y2-y1)).max(1.0);
                sum_x1 += x1 * w; sum_y1 += y1 * w; sum_x2 += x2 * w; sum_y2 += y2 * w; total_w += w;
            }
            Some((sum_x1/total_w, sum_y1/total_w, sum_x2/total_w, sum_y2/total_w))
        };

        let mut out: Vec<(f64,f64,f64,f64)> = Vec::new();
        if let Some(l) = average_line(&left_lines) { out.push(l); }
        if let Some(r) = average_line(&right_lines) { out.push(r); }
        Ok(out)
    })
}


