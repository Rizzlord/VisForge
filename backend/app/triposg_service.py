from __future__ import annotations

import asyncio
import base64
import gc
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
import trimesh
import tempfile

try:  # Optional dependency used for mesh simplification
    import pymeshlab  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional
    pymeshlab = None

import sys

REPO_ROOT = Path(__file__).resolve().parents[1] / "repos" / "TripoSG"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.image_process import load_image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

try:
    from scripts.briarmbg import BriaRMBG
except ModuleNotFoundError as exc:  # pragma: no cover - runtime safeguard
    raise ImportError(
        "The module 'scripts.briarmbg' could not be imported. Ensure the TripoSG repository"
        " under backend/repos/TripoSG contains the original 'scripts/briarmbg.py'."
    ) from exc

# Lazily imported in functions to avoid import-time cost when not used

MODELS_ROOT = Path(__file__).resolve().parents[1] / "repos" / "TripoSG" / "weights"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class TripoParams:
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


class TripoSGService:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: Optional[TripoSGPipeline] = None
        self._rmbg: Optional[BriaRMBG] = None
        self._lock = asyncio.Lock()
        self._current_dtype: Optional[torch.dtype] = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _download_models(self) -> tuple[Path, Path]:
        triposg_dir = MODELS_ROOT / "TripoSG"
        rmbg_dir = MODELS_ROOT / "RMBG-1.4"

        if not (triposg_dir / "model_index.json").exists():
            snapshot_download(
                repo_id="VAST-AI/TripoSG",
                local_dir=str(triposg_dir),
                local_dir_use_symlinks=False,
            )

        if not (rmbg_dir / "model.pth").exists():
            snapshot_download(
                repo_id="briaai/RMBG-1.4",
                local_dir=str(rmbg_dir),
                local_dir_use_symlinks=False,
            )

        return triposg_dir, rmbg_dir

    def _ensure_models(self, dtype: torch.dtype) -> None:
        if self._pipe is not None and self._rmbg is not None and self._current_dtype == dtype:
            return

        triposg_dir, rmbg_dir = self._download_models()

        self._rmbg = BriaRMBG.from_pretrained(str(rmbg_dir)).to(self._device)
        self._rmbg.eval()

        self._pipe = TripoSGPipeline.from_pretrained(str(triposg_dir)).to(self._device, dtype=dtype)
        self._current_dtype = dtype

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._rmbg is not None:
            del self._rmbg
            self._rmbg = None
        self._current_dtype = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _prepare_pil(self, image_data_url: str) -> Image.Image:
        if "base64," in image_data_url:
            _, encoded = image_data_url.split("base64,", 1)
        else:
            encoded = image_data_url
        image_bytes = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pil_image

    def _prepare_image_for_triposg(self, pil_image: Image.Image) -> Image.Image:
        assert self._rmbg is not None
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = Path(tmp.name)
            pil_image.save(temp_path)

        try:
            result = load_image(
                str(temp_path),
                bg_color=np.array([1.0, 1.0, 1.0]),
                rmbg_net=self._rmbg,
            )
            if isinstance(result, str):
                raise RuntimeError(result)
            img_np = result.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _simplify(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        if pymeshlab is None or mesh.faces.shape[0] <= target_faces:
            return mesh
        ms = pymeshlab.MeshSet()
        mesh_data = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
        ms.add_mesh(mesh_data)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
        )
        simplified = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=simplified.vertex_matrix(),
            faces=simplified.face_matrix(),
        )

    def generate(self, image_data_url: str, params: TripoParams) -> bytes:
        dtype = torch.float16 if params.use_float16 and self._device.type == "cuda" else torch.float32
        self._ensure_models(dtype)

        assert self._pipe is not None and self._rmbg is not None

        img_pil = self._prepare_pil(image_data_url)
        processed = self._prepare_image_for_triposg(img_pil)

        dense_depth = int(math.log2(params.dense_octree_resolution))
        hierarchical_depth = int(math.log2(params.hierarchical_octree_resolution)) + params.extra_depth_level
        flash_depth = int(math.log2(params.flash_octree_resolution))

        generator = torch.Generator(device=self._device).manual_seed(params.seed)
        output = self._pipe(
            image=processed,
            generator=generator,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.cfg_scale,
            use_flash_decoder=params.use_flash_decoder,
            dense_octree_depth=dense_depth,
            hierarchical_octree_depth=hierarchical_depth,
            flash_octree_depth=flash_depth,
        )

        meshes = output.meshes if hasattr(output, "meshes") else None
        if not meshes:
            raise RuntimeError("TripoSG pipeline returned empty mesh output")

        trimesh_mesh = meshes[0]

        if isinstance(trimesh_mesh, tuple):
            vertices, faces = trimesh_mesh
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
            trimesh_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.ascontiguousarray(faces))

        if params.simplify_mesh:
            trimesh_mesh = self._simplify(trimesh_mesh, params.target_face_number)

        glb_bytes = trimesh.exchange.gltf.export_glb(trimesh_mesh)

        if params.unload_model_after_generation:
            self.unload()
        else:
            gc.collect()
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        return glb_bytes


triposg_service = TripoSGService()
