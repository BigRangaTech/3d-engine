pub mod math;
pub mod scene_io;

use math::{Camera, Transform, Quat, Vec3};
use rayon::prelude::*;

#[derive(Clone)]
pub struct Entity {
    pub name: String,
    pub transform: Transform,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub mesh_path: String,
    pub is_character: bool,
    pub tag: String,
    pub layer: i32,
}

#[derive(Clone)]
pub struct Scene {
    pub entities: Vec<Entity>,
    pub camera: Camera,
    pub gravity: Vec3,
    pub linear_damping: f32,
}

pub struct Engine {
    pub scene: Scene,
    running: bool,
}

impl Engine {
    pub fn new(scene: Scene) -> Self {
        Self { scene, running: true }
    }

    pub fn update(&mut self, dt: f32) {
        // Simple physics: configurable gravity, per-entity acceleration, and linear damping.
        let gravity = self.scene.gravity;
        let damping = self.scene.linear_damping;

        self.scene
            .entities
            .par_iter_mut()
            .for_each(|entity| {
            // Net acceleration for this entity.
            let mut total_acc = gravity + entity.acceleration;

            // Extra CPU-heavy work for stress-test entities to better utilize
            // multiple cores. Entities with tag "stress" run additional math
            // each frame, which is safe because each entity is independent.
            if entity.tag == "stress" {
                let mut acc = total_acc;
                // Tunable iteration count: higher = heavier CPU load.
                for i in 0..2_000 {
                    let t = i as f32 * 0.001;
                    let s = (acc.x * t + acc.y).sin();
                    let c = (acc.y * t - acc.z).cos();
                    acc.x += s * 0.0001;
                    acc.y += c * 0.0001;
                    acc.z += (s * c) * 0.0001;
                }
                total_acc = acc;
            }

            if damping > 0.0 {
                total_acc -= entity.velocity * damping;
            }

            // Integrate velocity and position.
            entity.velocity += total_acc * dt;

            entity.transform.translation += entity.velocity * dt;

            // Ground plane at y = 0.0.
            if entity.transform.translation.y < 0.0 {
                entity.transform.translation.y = 0.0;
                if entity.velocity.y < 0.0 {
                    entity.velocity.y = 0.0;
                }
            }

            // Simple rotation around the Y axis for non-character entities.
            if !entity.is_character {
                let angle = 0.5 * dt;
                let rotation_step = Quat::from_rotation_y(angle);
                entity.transform.rotation = rotation_step * entity.transform.rotation;
            }
        });
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn shutdown(&mut self) {
        self.running = false;
    }
}
