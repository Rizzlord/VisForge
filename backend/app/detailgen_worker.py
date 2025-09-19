from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from skimage import measure

APP_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = APP_ROOT.parent
# The DetailGen3D repo lives at backend/repos/DetailGen3D (siblings of app)
REPO_ROOT = BACKEND_ROOT / "repos" / "DetailGen3D"
WEIGHTS_ROOT = REPO_ROOT / "pretrained_weights" / "DetailGen3D"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

# Ensure the backend root and repo paths are available for local imports.
for candidate in (APP_ROOT, BACKEND_ROOT, REPO_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from detailgen3d.inference_utils import generate_dense_grid_points  # noqa: E402
from detailgen3d.pipelines.pipeline_detailgen3d import DetailGen3DPipeline  # noqa: E402


def _decode_image(data_url: str) -> Image.Image:
    payload = data_url
    if "base64," in payload:
        _, payload = payload.split("base64,", 1)
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _load_mesh(model_base64: str) -> trimesh.Trimesh:
    data = base64.b64decode(model_base64)
    mesh = trimesh.load(io.BytesIO(data), file_type="glb")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Unsupported mesh format")
    return mesh


def _preprocess_mesh(input_mesh: trimesh.Trimesh, num_pc: int = 20480) -> torch.Tensor:
    mesh = input_mesh.copy()

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    if scale > 1e-6:
        mesh.apply_scale(1.9 / scale)

    try:
        surface, face_indices = trimesh.sample.sample_surface(mesh, 1_000_000)
        normal = mesh.face_normals[face_indices]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to sample surface from mesh. It might be invalid or have no faces."
        ) from exc

    if len(surface) == 0:
        raise RuntimeError("Mesh has no surface area to sample from.")

    num_pc_to_sample = min(num_pc, surface.shape[0])
    ind = np.random.default_rng().choice(surface.shape[0], num_pc_to_sample, replace=False)
    surface_tensor = torch.FloatTensor(surface[ind])
    normal_tensor = torch.FloatTensor(normal[ind])
    surface_tensor = torch.cat([surface_tensor, normal_tensor], dim=-1).unsqueeze(0)

    return surface_tensor


def _load_pipeline(device: torch.device, dtype: torch.dtype) -> DetailGen3DPipeline:
    local_dir = WEIGHTS_ROOT
    local_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DetailGen3DPipeline.from_pretrained(str(local_dir))
    return pipeline.to(device, dtype=dtype)


def _encode_mesh(mesh: trimesh.Trimesh) -> Tuple[str, int]:
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        export_path = Path(tmp.name)
    try:
        mesh.export(export_path, file_type="glb")
        data = export_path.read_bytes()
    finally:
        with contextlib.suppress(FileNotFoundError):
            export_path.unlink()
    return base64.b64encode(data).decode("utf-8"), len(data)


def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(payload["seed"])
    steps = int(payload["num_inference_steps"])
    guidance_scale = float(payload["guidance_scale"])
    noise_aug = float(payload.get("noise_aug", 0.0))

    pil_image = _decode_image(payload["image_data_url"])
    mesh = _load_mesh(payload["model_base64"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = _load_pipeline(device, dtype)

    surface = _preprocess_mesh(mesh).to(device, dtype=dtype)
    encoder_output = pipeline.vae.encode(surface).latent_dist.sample()

    box_min = np.array([-1.005, -1.005, -1.005])
    box_max = np.array([1.005, 1.005, 1.005])
    sampled_points, grid_size, bbox_size = generate_dense_grid_points(
        bbox_min=box_min, bbox_max=box_max, octree_depth=9, indexing="ij"
    )
    sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=dtype)
    sampled_points = sampled_points.unsqueeze(0)

    generator = torch.Generator(device=device).manual_seed(seed)

    sdf = pipeline(
        pil_image,
        latents=encoder_output,
        sampled_points=sampled_points,
        noise_aug_level=noise_aug,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).samples[0]

    grid_logits = sdf.view(grid_size).cpu().numpy()
    vertices, faces, normals, _ = measure.marching_cubes(grid_logits, 0, method="lewiner")
    vertices = vertices / grid_size * bbox_size + box_min
    output_mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))

    glb_base64, size = _encode_mesh(output_mesh)

    # Optionally unload models and clear GPU memory to free VRAM
    try:
        if bool(payload.get("unload_model_after_generation", False)):
            # Delete large objects and pipeline references
            try:
                del pipeline
            except Exception:
                pass

            # Delete intermediate tensors if they exist
            for name in ("encoder_output", "sampled_points", "surface"):
                try:
                    if name in locals():
                        del locals()[name]
                except Exception:
                    pass

            # Force Python GC and clear CUDA cache if available
            import gc as _gc

            _gc.collect()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    except Exception:
        # Ensure we don't mask the success response if cleanup fails
        pass

    return {
        "glb_base64": glb_base64,
        "file_name": payload.get("file_name", "detailgen-refined.glb"),
        "mime_type": "model/gltf-binary",
        "size": size,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    result = process(payload)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    import contextlib  # noqa: E402

    main()
