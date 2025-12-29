use std::collections::HashMap;

const Focal_Length: f64 = 700.0;
const Real_Car_width: f64 = 1.8; //Avg car width

#[derive(Clone, Debug)]
pub struct TrackedObject {
    pub id: usize,
    pub bbox: (f64, f64, f64, f64),
    pub distance: f64, // m
    pub speed: f64,    // m/s
    pub collisiontime: f64, //s
    pub lost_frames: usize, // old objects
}

pub struct ObjectTracker {
    pub nect_id: usize, // restored your name
    pub objects: HashMap<usize, TrackedObject>,
}

impl ObjectTracker {
    pub fn new() -> Self {
        ObjectTracker {
            nect_id: 0,
            objects: HashMap::new(), // Fixed typo: HasMap -> HashMap
        }
    }

    fn calc_distancee(bbox_width: f64) -> f64 { // restored your name
        if bbox_width <= 1.0 {
            return 100.0;
        }
        (Focal_Length * Real_Car_width) / bbox_width
    }

    pub fn process_frame(&mut self, detections: Vec<(f64, f64, f64, f64)>, dt: f64) -> Vec<TrackedObject> {
        let mut new_objects: HashMap<usize, TrackedObject> = HashMap::new(); // Fixed typo: Hashmap -> HashMap
        
        for (x, y, w, h) in detections {
            let cx = x + w / 2.0;
            let cy = y + h / 2.0;
            let current_dist = Self::calc_distancee(w);

            let mut best_match = None;
            let mut min_error = f64::MAX;
            
            // Currently uses centroid matching as a robust method for building tracker
            for (id, old_obj) in &self.objects {
                let (ox, oy, ow, oh) = old_obj.bbox;
                let ocx = ox + ow / 2.0;
                let ocy = oy + oh / 2.0;

                // Fixed syntax: (cx - ocx)
                let error = (cx - ocx).powi(2) + (cy - ocy).powi(2);

                if error < 10000.0 && error < min_error {
                    min_error = error;
                    best_match = Some(*id);
                }
            }

            if let Some(id) = best_match {
                if let Some(prev_obj) = self.objects.get(&id) {
                    let speed = (prev_obj.distance - current_dist) / dt;

                    let collisiontime = if speed > 0.1 {
                        current_dist / speed
                    } else {
                        99.0 // Removed semicolon to allow assignment
                    };

                    new_objects.insert(id, TrackedObject {
                        id,
                        bbox: (x, y, w, h),
                        distance: current_dist,
                        speed,
                        collisiontime, // Fixed period to comma
                        lost_frames: 0,
                    });
                }
                self.objects.remove(&id);
            } else {
                // creating new_object
                let id = self.nect_id; 
                self.nect_id += 1; // Fixed increment syntax

                new_objects.insert(id, TrackedObject {
                    id,
                    bbox: (x, y, w, h),
                    distance: current_dist, // Corrected from curr_dist
                    speed: 0.0,
                    collisiontime: 99.0, // Corrected from ttc to match struct
                    lost_frames: 0,
                });
            }
        }

        self.objects = new_objects;
        self.objects.values().cloned().collect()
    }
}