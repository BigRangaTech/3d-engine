use engine::Scene;
use engine::math::Vec3;

use crate::InputState;

pub struct CameraOrbitState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

pub fn update_player_from_input(scene: &mut Scene, input: &InputState, dt: f32) {
    let player_opt = scene
        .entities
        .iter_mut()
        .find(|e| e.tag == "player");
    let Some(player) = player_opt else {
        return;
    };

    let mut move_dir = Vec3::ZERO;
    if input.move_forward {
        move_dir.z -= 1.0;
    }
    if input.move_back {
        move_dir.z += 1.0;
    }
    if input.move_left {
        move_dir.x -= 1.0;
    }
    if input.move_right {
        move_dir.x += 1.0;
    }

    if move_dir.length_squared() > 0.0 {
        let speed = 5.0;
        let delta = move_dir.normalize() * speed * dt;
        player.transform.translation += delta;
    }
}

pub fn check_triggers(scene: &Scene) {
    let player_pos = match scene.entities.iter().find(|e| e.tag == "player") {
        Some(p) => p.transform.translation,
        None => return,
    };

    for trigger in scene.entities.iter().filter(|e| e.tag == "trigger") {
        let center = trigger.transform.translation;
        let half_extents = trigger.transform.scale * 0.5;
        let delta = player_pos - center;
        if delta.x.abs() <= half_extents.x
            && delta.y.abs() <= half_extents.y
            && delta.z.abs() <= half_extents.z
        {
            log::info!("Player entered trigger '{}'", trigger.name);
        }
    }
}

pub fn update_third_person_camera(scene: &mut Scene, orbit: &CameraOrbitState) {
    let player_pos = match scene.entities.iter().find(|e| e.tag == "player") {
        Some(p) => p.transform.translation,
        None => return,
    };

    let yaw = orbit.yaw;
    let pitch = orbit.pitch;
    let distance = orbit.distance.max(0.1);

    let cp = pitch.cos();
    let sp = pitch.sin();
    let cy = yaw.cos();
    let sy = yaw.sin();

    let dir = Vec3::new(sy * cp, sp, cy * cp);

    let offset = dir * distance + Vec3::new(0.0, 1.0, 0.0);
    scene.camera.position = player_pos + offset;
    scene.camera.target = player_pos;
}
