from __future__ import annotations

import base64
import contextlib
import gc
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
import trimesh

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')


@dataclass
class WorkerParams:
    seed: int
    use_float16: bool
    extra_depth_level: int
    num_inference_steps: int
    cfg_scale: float
    simplify_mesh: bool
    target_face_number: int
    use_flash_decoder: bool
    dense_octree_resolution: int
    hierarchical_octree_resolution: int
    flash_octree_resolution: int
    unload_model_after_generation: bool


def ensure_paths(paths: list[str]) -> None:
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def load_image_module(scripts_root: Path):
    ensure_paths([str(scripts_root)])
    from scripts.image_process import load_image
    return load_image


def load_briarmbg_module(scripts_root: Path):
    ensure_paths([str(scripts_root)])
    from scripts.briarmbg import BriaRMBG
    return BriaRMBG


def download_weights(weights_root: Path) -> tuple[Path, Path]:
    triposg_dir = weights_root / 'TripoSG'
    rmbg_dir = weights_root / 'RMBG-1.4'

    if not (triposg_dir / 'model_index.json').exists():
        snapshot_download(
            repo_id='VAST-AI/TripoSG',
            local_dir=str(triposg_dir),
            local_dir_use_symlinks=False,
        )

    if not (rmbg_dir / 'model.pth').exists():
        snapshot_download(
            repo_id='briaai/RMBG-1.4',
            local_dir=str(rmbg_dir),
            local_dir_use_symlinks=False,
        )

    return triposg_dir, rmbg_dir


def prepare_image(pil_image: Image.Image, scripts_root: Path, rmbg_net, device: torch.device) -> Image.Image:
    load_image = load_image_module(scripts_root)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = Path(tmp.name)
        pil_image.save(temp_path)

    try:
        tensor_or_error = load_image(
            str(temp_path),
            bg_color=np.array([1.0, 1.0, 1.0]),
            rmbg_net=rmbg_net,
        )
        if isinstance(tensor_or_error, str):
            raise RuntimeError(tensor_or_error)
        tensor = tensor_or_error.detach().cpu().permute(1, 2, 0).numpy()
        tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(tensor)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    weights_root = Path(payload['weights_root'])
    scripts_root = Path(payload['scripts_root'])
    repo_root = Path(payload['repo_root'])

    ensure_paths([str(repo_root)])

    triposg_dir, rmbg_dir = download_weights(weights_root)

    BriaRMBG = load_briarmbg_module(scripts_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if payload['use_float16'] and device.type == 'cuda' else torch.float32

    rmbg_net = BriaRMBG.from_pretrained(str(rmbg_dir)).to(device)
    rmbg_net.eval()

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(str(triposg_dir)).to(device, dtype=dtype)

    image_data = payload['image_data_url']
    if 'base64,' in image_data:
        _, image_data = image_data.split('base64,', 1)
    pil_image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    processed = prepare_image(pil_image, scripts_root, rmbg_net, device)

    generator = torch.Generator(device=device).manual_seed(payload['seed'])

    dense_depth = int(math.log2(payload['dense_octree_resolution']))
    hierarchical_depth = int(math.log2(payload['hierarchical_octree_resolution'])) + payload['extra_depth_level']
    flash_depth = int(math.log2(payload['flash_octree_resolution']))

    output = pipe(
        image=processed,
        generator=generator,
        num_inference_steps=payload['num_inference_steps'],
        guidance_scale=payload['cfg_scale'],
        use_flash_decoder=payload['use_flash_decoder'],
        dense_octree_depth=dense_depth,
        hierarchical_octree_depth=hierarchical_depth,
        flash_octree_depth=flash_depth,
    )

    meshes = output.meshes if hasattr(output, 'meshes') else None
    if not meshes:
        raise RuntimeError('TripoSG pipeline returned empty mesh output')

    trimesh_mesh = meshes[0]
    if isinstance(trimesh_mesh, tuple):
        vertices, faces = trimesh_mesh
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.ascontiguousarray(faces))

    if payload['simplify_mesh'] and payload['target_face_number'] > 0:
        try:
            import pymeshlab  # type: ignore

            ms = pymeshlab.MeshSet()
            mesh_data = pymeshlab.Mesh(vertex_matrix=trimesh_mesh.vertices, face_matrix=trimesh_mesh.faces)
            ms.add_mesh(mesh_data)
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=payload['target_face_number'],
                preserveboundary=True,
            )
            simplified = ms.current_mesh()
            trimesh_mesh = trimesh.Trimesh(
                vertices=simplified.vertex_matrix(),
                faces=simplified.face_matrix(),
            )
        except ModuleNotFoundError:
            pass

    glb_bytes = trimesh.exchange.gltf.export_glb(trimesh_mesh)

    if payload['unload_model_after_generation']:
        del pipe
        del rmbg_net
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'glb_base64': base64.b64encode(glb_bytes).decode('utf-8'),
        'file_name': 'tripo-model.glb',
        'mime_type': 'model/gltf-binary',
    }


def main() -> None:
    raw_input = sys.stdin.read()
    payload = json.loads(raw_input)
    with contextlib.redirect_stdout(sys.stderr):
        result = run_worker(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == '__main__':
    main()
