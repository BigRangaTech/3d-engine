pub mod math;
pub mod scene_io;

use math::{Camera, Transform, Quat, Vec3};

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

        for entity in &mut self.scene.entities {
            // Net acceleration for this entity.
            let mut total_acc = gravity + entity.acceleration;
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
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn shutdown(&mut self) {
        self.running = false;
    }
}
