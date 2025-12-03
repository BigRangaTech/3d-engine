pub mod math;

use math::{Camera, Transform, Quat, Vec3};

pub struct Entity {
    pub name: String,
    pub transform: Transform,
    pub velocity: Vec3,
}

pub struct Scene {
    pub entities: Vec<Entity>,
    pub camera: Camera,
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
        // Very simple "systems": gravity + rotation + integration.
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        for entity in &mut self.scene.entities {
            // Apply gravity.
            entity.velocity += gravity * dt;

            // Integrate velocity into position.
            entity.transform.translation += entity.velocity * dt;

            // Ground plane at y = 0.0.
            if entity.transform.translation.y < 0.0 {
                entity.transform.translation.y = 0.0;
                if entity.velocity.y < 0.0 {
                    entity.velocity.y = 0.0;
                }
            }

            // Simple rotation around the Y axis.
            let angle = 0.5 * dt;
            let rotation_step = Quat::from_rotation_y(angle);
            entity.transform.rotation = rotation_step * entity.transform.rotation;
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn shutdown(&mut self) {
        self.running = false;
    }
}
