use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use log::{error, info, warn};

use image::io::Reader as ImageReader;
use image::ImageFormat;

use egui::{ColorImage, Context as EguiContext, TextureHandle, TextureOptions};

use crate::MeshAsset;

const MAX_MESH_ASSETS: usize = 512;
const MAX_THUMBNAILS: usize = 128;

pub fn discover_mesh_assets(ctx: &EguiContext, root: &str) -> Vec<MeshAsset> {
    let mut result = Vec::new();
    let root_path = Path::new(root);
    let mut thumbs_loaded = 0usize;
    collect_mesh_assets(ctx, root_path, root_path, &mut result, &mut thumbs_loaded);
    result
}

pub fn convert_assets_to_glb(root: &str, dst_root: &str) {
    let root_path = Path::new(root);
    let dst_root_path = Path::new(dst_root);

    if !root_path.exists() {
        error!("Asset root '{}' does not exist", root);
        return;
    }

    let mut fbx_available = true;
    convert_dir(root_path, root_path, dst_root_path, &mut fbx_available);
}

fn convert_dir(dir: &Path, root: &Path, dst_root: &Path, fbx_available: &mut bool) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            error!("Failed to read dir '{}': {}", dir.display(), e);
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            convert_dir(&path, root, dst_root, fbx_available);
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "glb" || ext == "gltf" || ext == "obj" || ext == "stl" {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let dst = dst_root.join(rel);
            if let Some(parent) = dst.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    error!(
                        "Failed to create destination dir '{}': {}",
                        parent.display(),
                        e
                    );
                    continue;
                }
            }
            if dst.exists() {
                continue;
            }
            match fs::copy(&path, &dst) {
                Ok(_) => info!("Copied asset {} -> {}", path.display(), dst.display()),
                Err(e) => error!(
                    "Failed to copy asset {} -> {}: {}",
                    path.display(),
                    dst.display(),
                    e
                ),
            }
        } else if ext == "fbx" && *fbx_available {
            convert_fbx(&path, root, dst_root, fbx_available);
        }
    }
}

fn convert_fbx(path: &Path, root: &Path, dst_root: &Path, fbx_available: &mut bool) {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let out_dir = dst_root.join(rel).with_extension("");
    if let Some(parent) = out_dir.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            error!(
                "Failed to create FBX output dir '{}': {}",
                parent.display(),
                e
            );
            return;
        }
    }

    let status = Command::new("fbx2gltf")
        .arg("-i")
        .arg(path)
        .arg("-o")
        .arg(&out_dir)
        .status();

    match status {
        Ok(s) if s.success() => {
            info!("Converted FBX {} -> {}", path.display(), out_dir.display());
        }
        Ok(s) => {
            error!(
                "fbx2gltf failed for {} with status: {}",
                path.display(),
                s
            );
        }
        Err(e) => {
            warn!(
                "fbx2gltf not available or failed to start ({}); skipping further FBX conversions",
                e
            );
            *fbx_available = false;
        }
    }
}

fn is_mesh_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            ext == "glb" || ext == "gltf" || ext == "obj" || ext == "stl"
        })
        .unwrap_or(false)
}

fn collect_mesh_assets(
    ctx: &EguiContext,
    dir: &Path,
    root: &Path,
    out: &mut Vec<MeshAsset>,
    thumbs_loaded: &mut usize,
) {
    if out.len() >= MAX_MESH_ASSETS {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if out.len() >= MAX_MESH_ASSETS {
            break;
        }

        let path = entry.path();
        if path.is_dir() {
            // Skip obviously non-mesh-heavy folders to keep startup fast on large packs.
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let lower = name.to_ascii_lowercase();
                if lower == "textures"
                    || lower == "texture"
                    || lower == "preview"
                    || lower == "previews"
                    || lower == "source"
                    || lower == "sources"
                    || lower == "animations"
                    || lower == "skins"
                {
                    continue;
                }
            }
            collect_mesh_assets(ctx, &path, root, out, thumbs_loaded);
        } else if is_mesh_file(&path) {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let name = rel.to_string_lossy().to_string();

            let thumbnail = if *thumbs_loaded < MAX_THUMBNAILS {
                let tex = find_preview_image_for_mesh(&path, root)
                    .and_then(|img_path| load_texture_from_png(ctx, &img_path).ok());
                if tex.is_some() {
                    *thumbs_loaded += 1;
                }
                tex
            } else {
                None
            };

            out.push(MeshAsset {
                name,
                mesh_path: path.to_string_lossy().to_string(),
                thumbnail,
            });
        }
    }
}

fn find_preview_image(dir: &Path) -> Option<PathBuf> {
    // Prefer a file literally named Preview.png in this directory,
    // otherwise the first .png found in this directory.
    let entries = fs::read_dir(dir).ok()?;

    let mut first_png: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
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
            if first_png.is_none() {
                first_png = Some(path);
            }
        }
    }

    first_png
}

fn find_preview_image_for_mesh(mesh_path: &Path, root: &Path) -> Option<PathBuf> {
    // Walk up the directory tree from the mesh file's directory toward the
    // asset root, looking for a suitable preview image in each directory.
    let mut current = mesh_path.parent();
    while let Some(dir) = current {
        if let Some(img) = find_preview_image(dir) {
            return Some(img);
        }
        if dir == root {
            break;
        }
        current = dir.parent();
    }
    None
}

fn load_texture_from_png(ctx: &EguiContext, path: &Path) -> Result<TextureHandle, String> {
    let file = fs::File::open(path).map_err(|e| format!("open error: {e}"))?;
    let mut reader = ImageReader::new(std::io::BufReader::new(file));
    reader.set_format(ImageFormat::Png);
    let image = reader.decode().map_err(|e| format!("decode error: {e}"))?;
    let size = [image.width() as usize, image.height() as usize];
    let rgba = image.into_rgba8();
    let color_image = ColorImage::from_rgba_unmultiplied(size, &rgba);
    Ok(ctx.load_texture(
        path.to_string_lossy(),
        color_image,
        TextureOptions::LINEAR,
    ))
}
