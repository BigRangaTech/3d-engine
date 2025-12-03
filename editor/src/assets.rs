use std::fs;
use std::path::{Path, PathBuf};

use egui_extras::RetainedImage;

use crate::MeshAsset;

pub fn discover_mesh_assets(root: &str) -> Vec<MeshAsset> {
    let mut result = Vec::new();
    let root_path = Path::new(root);
    let entries = match fs::read_dir(root_path) {
        Ok(e) => e,
        Err(_) => return result,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = entry
            .file_name()
            .to_string_lossy()
            .to_string();

        if let Some(glb_path) = find_first_glb(&path) {
            let thumbnail = find_preview_image(&path)
                .and_then(|img_path| fs::read(&img_path).ok())
                .and_then(|bytes| {
                    RetainedImage::from_image_bytes(
                        img_path.to_string_lossy(),
                        &bytes,
                    )
                    .ok()
                });

            result.push(MeshAsset {
                name,
                glb_path: glb_path.to_string_lossy().to_string(),
                thumbnail,
            });
        }
    }

    result
}

fn find_first_glb(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_first_glb(&path) {
                return Some(found);
            }
        } else if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("glb"))
            .unwrap_or(false)
        {
            return Some(path);
        }
    }
    None
}

fn find_preview_image(dir: &Path) -> Option<PathBuf> {
    // Prefer a file literally named Preview.png, otherwise the first .png found.
    let mut candidate: Option<PathBuf> = None;
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if candidate.is_none() {
                candidate = find_preview_image(&path);
            }
            continue;
        }

        if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
            if fname.eq_ignore_ascii_case("preview.png") {
                return Some(path);
            }
        }

        if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("png"))
            .unwrap_or(false)
        {
            if candidate.is_none() {
                candidate = Some(path);
            }
        }
    }

    candidate
}

