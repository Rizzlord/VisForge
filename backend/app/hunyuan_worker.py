from __future__ import annotations

import base64
import contextlib
import gc
import io
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
import trimesh

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@dataclass
class WorkerParams:
    seed: int
    randomize_seed: bool
    remove_background: bool
    num_inference_steps: int
    guidance_scale: float
    octree_resolution: int
    num_chunks: int
    mc_algo: str
    unload_model_after_generation: bool


def ensure_paths(paths: list[str]) -> None:
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def apply_torchvision_fix(repo_root: Path) -> None:
    try:
        ensure_paths([str(repo_root)])
        from torchvision_fix import apply_fix  # type: ignore

        apply_fix()
    except ImportError:
        pass
    except Exception:
        pass


def download_weights(weights_root: Path) -> Path:
    target_dir = weights_root / "tencent-Hunyuan3D-2.1"
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2.1",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def decode_image(image_data_url: str) -> Image.Image:
    image_data = image_data_url
    if "base64," in image_data:
        _, image_data = image_data.split("base64,", 1)
    raw = base64.b64decode(image_data)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    weights_root = Path(payload["weights_root"])
    repo_root = Path(payload["repo_root"])

    ensure_paths([str(repo_root), str(repo_root / "hy3dshape"), str(repo_root / "hy3dpaint")])
    apply_torchvision_fix(repo_root)

    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
    from hy3dshape.pipelines import export_to_trimesh  # type: ignore
    from hy3dshape.rembg import BackgroundRemover  # type: ignore

    params = WorkerParams(
        seed=int(payload["seed"]),
        randomize_seed=bool(payload["randomize_seed"]),
        remove_background=bool(payload["remove_background"]),
        num_inference_steps=int(payload["num_inference_steps"]),
        guidance_scale=float(payload["guidance_scale"]),
        octree_resolution=int(payload["octree_resolution"]),
        num_chunks=int(payload["num_chunks"]),
        mc_algo=str(payload["mc_algo"]).lower(),
        unload_model_after_generation=bool(payload["unload_model_after_generation"]),
    )

    if params.mc_algo not in {"mc", "dmc"}:
        raise RuntimeError(f"Unsupported surface extraction algorithm: {params.mc_algo}")

    model_dir = download_weights(weights_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(model_dir),
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        device=device.type,
    )
    pipeline.to(device)

    remover = BackgroundRemover()

    image = decode_image(payload["image_data_url"])
    if params.remove_background or image.mode == "RGB":
        image = remover(image.convert("RGB"))

    final_seed = params.seed
    if params.randomize_seed:
        final_seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=device).manual_seed(int(final_seed))

    try:
        outputs = pipeline(
            image=image,
            generator=generator,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            octree_resolution=params.octree_resolution,
            num_chunks=params.num_chunks,
            mc_algo=params.mc_algo,
            output_type="mesh",
        )
    except ImportError as exc:
        raise RuntimeError(str(exc)) from exc

    meshes = export_to_trimesh(outputs)

    mesh = None
    if isinstance(meshes, list):
        for candidate in meshes:
            if candidate is None:
                continue
            mesh = candidate
            break
    else:
        mesh = meshes

    if mesh is None:
        raise RuntimeError("Hunyuan3D pipeline returned empty mesh output")

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    glb_bytes = trimesh.exchange.gltf.export_glb(mesh)

    if params.unload_model_after_generation:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
        "file_name": "hunyuan-model.glb",
        "mime_type": "model/gltf-binary",
        "seed": int(final_seed),
    }


def main() -> None:
    raw_input = sys.stdin.read()
    payload = json.loads(raw_input)
    with contextlib.redirect_stdout(sys.stderr):
        result = run_worker(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
