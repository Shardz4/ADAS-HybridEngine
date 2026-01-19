use ndarray::ArrayView3;

#[derive(Debug, Clone, Copy)]
pub enum LightStatus {
    Green,
    Yellow,
    Red,
    None,
}

pub fn detect_traffic_light(frame: &ArrayView3<u8>) -> LightStatus {
    let (h, w, _) = frame.dim();
    let roi_height = h / 3;

    let mut red_count = 0;
    let mut yellow_count = 0;
    let mut green_count = 0;

    for y in 0..roi_height {
        for x in 0..w {
            let r = frame[[y, x, 0]] as f32;
            let g = frame[[y, x, 1]] as f32; 
            let b = frame[[y, x, 2]] as f32;

            if r < 150.0 && g < 150.0 && b < 150.0 {
                continue;
            }

            // rgb to hsv conversion
            let max = r.max(g).max(b);
            let min = r.min(g).min(b); 
            let delta = max - min;

           
            if max == 0.0 || delta < 20.0 {
                continue;
            }

            let mut h = if max == r {
                (g - b) / delta
            } else if max == g {
                2.0 + (b - r) / delta
            } else {
                4.0 + (r - g) / delta
            };

            h *= 60.0;
            if h < 0.0 {
                h += 360.0;
            }

            // Classifying colors
            if h <= 15.0 || h >= 345.0 {
                red_count += 1;
            } else if h > 15.0 && h <= 35.0 {
                yellow_count += 1;
            } else if h > 40.0 && h <= 90.0 {
                green_count += 1;
            }
        }
    }

    let threshold = (roi_height * w) / 100;

    if red_count > threshold && red_count > green_count && red_count > yellow_count {
        return LightStatus::Red;
    } 
    if yellow_count > threshold && yellow_count > red_count && yellow_count > green_count {
        return LightStatus::Yellow;
    }
    if green_count > threshold && green_count > red_count && green_count > yellow_count {
        return LightStatus::Green;
    }

    LightStatus::None
}