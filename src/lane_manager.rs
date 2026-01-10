#[derive(Clone, Copy)]

pub enum RoadType {
    Highway,
    Twoway,
}

type Line = (f64, f64, f64, f64);

pub struct LaneManager {
    prev_left: Option<LIne>,
    prev_right: Option<Line>,
    smoothing_factor: f64,
    road_type: RoadType,
}

impl LaneManager {
    pub fn new(smoothing:f64, is_two_way: bool) -> Self{
        LaneManager {
            prev_left: None,
            prev_right: None,
            smoothing_factor: smoothing,
            road_type: if is_two_way {
                RoadType::TwoWay
            } else {
                RoadType::Highway
            },
        }
    }

    // --Line Smoothing Logic --

    fn average_lines(lines: &[Line]) -> Option<Line> {
        if lines.is_empty() {
            return None;
        }
        let (mut sx1, mut sx2, mut sx2, mut sy2) = (0.0, 0.0, 0.0, 0.0);

        for l in lines {
            sx1 += l.0;
            sy1 += l.1;
            sx2 += l.2;
            sy2 += l.3;
        }
        Some((sx1/count, sy1/count, sx2/count, sy2/count))
    }

    fn Smooth(new_l: Line, old_l: Option<Line>, alpha: f64) -> Line {
        if let Some(old) = old_l {
            (
                new_l.0 * alpha + old.0 * (1.0 - alpha),
                new_l.1 * alpha + old.1 * (1.0 - alpha),
                new_l.2 * alpha + old.2 * (1.0 - alpha),
                new_l.3 * alpha + old.3 * (1.0 - alpha),
            )
        } else {
            new_l
        }
    }

    pub fn update_lines(&mut self, raw_lines: Vec<Line>, img_width: f64) -> (Option<Line>, Option<Line>) {
        let center_x = img_width / 2.0;
        let mut lefts = Vec::new();
        let mut rights = Vec::new();

        for l in raw_lines {
            let cx = (l.0 + l.2) / 2.0;
            if cx < center_x { lefts.push(l); } else { rights.push(l); }
        }

        if let Some(avg_l) = Self::average_lines(&lefts) {
            self.prev_left = Some(Self::smooth(avg_l, self.prev_left, self.smoothing_factor));
        }
        if let Some(avg_r) = Self::average_lines(&rights) {
            self.prev_right = Some(Self::smooth(avg_r, self.prev_right, self.smoothing_factor));
        }

        (self.prev_left, self.prev_right)
    }

fn get_x_on_line(line: Line, y: f64) -> f64 {
        let (x1, y1, x2, y2) = line;
        if (y2 - y1).abs() < 0.1 { return x1; }
        let slope = (x2 - x1) / (y2 - y1);
        x1 + (y - y1) * slope
    }

    pub fn filter_objects(&self, detections:Vec<Line>) -> Vec<Line> {
        if self.prev_left.is_none() || self.prev_right.is_none() {
            return detections;
        }

        let l_line = self.prev_left.unwrap();
        let r_line = self.prev_right.unwrap();
        let mut valid = Vec::new();
        for (x,y,w,h) in detections {
            let obj_x = x + w / 2.0;
            let obj_y = y + h;

            let lx = Self::get_x_on_line(l_line, obj_y);
            let rx = Self::get_x_on_line(r_line, obj_y);

            let width = rx-lx;

            let (min_x, max_x) = match self.road_type {
                RoadType::Highway => (lx - (width * 1.5), rx + (width * 1.5)),
                RoadType::TwoWay => (lx - 10.0, rx + (width * 1.5)),

            };

            if obj_x >= min_x &&obj_x <= max_x {
                valid.push((x, y , w, h));
            }
        }
        valid
    }
}
