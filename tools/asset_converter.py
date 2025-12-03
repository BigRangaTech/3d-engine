#!/usr/bin/env python3
"""
Simple asset converter for the Rust 3D engine.

Usage examples (run from workspace root):

  python tools/asset_converter.py editor/3D assets converted
  python tools/asset_converter.py path/to/file.fbx converted

Behavior:
  - Copies supported formats as‑is: .glb, .gltf, .obj, .stl
  - Optionally converts .fbx -> .glb using the external `fbx2gltf` tool if available.
    You must install `fbx2gltf` yourself and ensure it is on PATH.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


SUPPORTED_DIRECT = {".glb", ".gltf", ".obj", ".stl"}
CONVERT_FBX = {".fbx"}


def find_fbx2gltf() -> Path | None:
    exe = shutil.which("fbx2gltf")
    return Path(exe) if exe else None


def copy_asset(src: Path, dst_root: Path, input_root: Path) -> None:
    rel = src.relative_to(input_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")


def convert_fbx(src: Path, dst_root: Path, input_root: Path, fbx2gltf: Path) -> None:
    rel = src.relative_to(input_root)
    out_dir = (dst_root / rel).with_suffix("")  # fbx2gltf writes into a folder
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(fbx2gltf),
        "-i",
        str(src),
        "-o",
        str(out_dir),
        "--khr-materials-unlit",
    ]
    print(f"[fbx2gltf] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] fbx2gltf failed for {src}: {e}")


def process_path(path: Path, dst_root: Path) -> None:
    if path.is_file():
        input_root = path.parent
        paths = [path]
    else:
        input_root = path
        paths = [p for p in path.rglob("*") if p.is_file()]

    fbx2gltf = find_fbx2gltf()
    if any(p.suffix.lower() in CONVERT_FBX for p in paths) and fbx2gltf is None:
        print(
            "[warn] FBX files detected but `fbx2gltf` is not available on PATH; "
            "FBX files will be skipped."
        )

    for src in paths:
        ext = src.suffix.lower()
        if ext in SUPPORTED_DIRECT:
            copy_asset(src, dst_root, input_root)
        elif ext in CONVERT_FBX and fbx2gltf is not None:
            convert_fbx(src, dst_root, input_root, fbx2gltf)
        else:
            # Skip other formats; they may need manual conversion (Blender, etc.).
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert/copy 3D assets for the Rust engine.")
    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory (e.g. 'editor/3D assets')",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("converted_assets"),
        help="Output directory (default: converted_assets)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")

    process_path(args.input, args.output)


if __name__ == "__main__":
    main()

