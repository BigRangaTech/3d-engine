use std::fs;
use std::path::{Path, PathBuf};

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
