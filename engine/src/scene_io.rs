use std::fs;
use std::path::Path;

use crate::{math, Camera, Entity, Scene};
use math::{Quat, Vec3};

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableTransform {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableEntity {
    name: String,
    transform: SerializableTransform,
    velocity: [f32; 3],
    #[serde(default)]
    acceleration: Option<[f32; 3]>,
    mesh_path: String,
    #[serde(default)]
    is_character: bool,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    layer: Option<i32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableCamera {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    fov_y_radians: f32,
    aspect: f32,
    z_near: f32,
    z_far: f32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableScene {
    entities: Vec<SerializableEntity>,
    camera: SerializableCamera,
    #[serde(default)]
    gravity: Option<[f32; 3]>,
    #[serde(default)]
    linear_damping: Option<f32>,
}

fn scene_to_serializable(scene: &Scene) -> SerializableScene {
    let entities = scene
        .entities
        .iter()
        .map(|e| SerializableEntity {
            name: e.name.clone(),
            transform: SerializableTransform {
                translation: e.transform.translation.to_array(),
                rotation: {
                    let q = e.transform.rotation;
                    [q.x, q.y, q.z, q.w]
                },
                scale: e.transform.scale.to_array(),
            },
            velocity: e.velocity.to_array(),
            acceleration: Some(e.acceleration.to_array()),
            mesh_path: e.mesh_path.clone(),
            is_character: e.is_character,
            tag: Some(e.tag.clone()),
            layer: Some(e.layer),
        })
        .collect();

    let cam = &scene.camera;
    let camera = SerializableCamera {
        position: cam.position.to_array(),
        target: cam.target.to_array(),
        up: cam.up.to_array(),
        fov_y_radians: cam.fov_y_radians,
        aspect: cam.aspect,
        z_near: cam.z_near,
        z_far: cam.z_far,
    };

    SerializableScene {
        entities,
        camera,
        gravity: Some(scene.gravity.to_array()),
        linear_damping: Some(scene.linear_damping),
    }
}

fn serializable_to_scene(data: &SerializableScene) -> Scene {
    let entities = data
        .entities
        .iter()
        .map(|e| Entity {
            name: e.name.clone(),
            transform: math::Transform {
                translation: Vec3::from_array(e.transform.translation),
                rotation: Quat::from_xyzw(
                    e.transform.rotation[0],
                    e.transform.rotation[1],
                    e.transform.rotation[2],
                    e.transform.rotation[3],
                ),
                scale: Vec3::from_array(e.transform.scale),
            },
            velocity: Vec3::from_array(e.velocity),
            acceleration: e
                .acceleration
                .map(Vec3::from_array)
                .unwrap_or(Vec3::ZERO),
            mesh_path: e.mesh_path.clone(),
            is_character: e.is_character,
            tag: e.tag.clone().unwrap_or_default(),
            layer: e.layer.unwrap_or(0),
        })
        .collect();

    let cam = &data.camera;
    let camera = Camera {
        position: Vec3::from_array(cam.position),
        target: Vec3::from_array(cam.target),
        up: Vec3::from_array(cam.up),
        fov_y_radians: cam.fov_y_radians,
        aspect: cam.aspect,
        z_near: cam.z_near,
        z_far: cam.z_far,
    };
    let gravity = data
        .gravity
        .map(Vec3::from_array)
        .unwrap_or_else(|| Vec3::new(0.0, -9.81, 0.0));
    let linear_damping = data.linear_damping.unwrap_or(0.2);

    Scene {
        entities,
        camera,
        gravity,
        linear_damping,
    }
}

pub fn save_scene_to_file(path: &str, scene: &Scene) -> Result<(), String> {
    let serializable = scene_to_serializable(scene);
    let json = serde_json::to_string_pretty(&serializable)
        .map_err(|e| format!("serialize error: {e}"))?;
    fs::write(Path::new(path), json).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

pub fn load_scene_from_file(path: &str) -> Option<Scene> {
    let data = fs::read_to_string(Path::new(path)).ok()?;
    let serializable: SerializableScene = serde_json::from_str(&data).ok()?;
    Some(serializable_to_scene(&serializable))
}
