from __future__ import annotations

import base64
import contextlib
import gc
import io
import json
import logging
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TextureWorkerParams:
    seed: int
    randomize_seed: bool
    max_view_count: int
    view_resolution: int
    num_inference_steps: int
    guidance_scale: float
    target_face_count: int
    remesh_mesh: bool
    decimate: bool
    uv_unwrap: bool
    texture_resolution: int
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


def configure_caches(weights_root: Path) -> None:
    cache_dir = weights_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(weights_root / "torch"))


def ensure_paint_weights(weights_root: Path) -> Path:
    target_dir = weights_root / "tencent-Hunyuan3D-2.1"
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ensuring Hunyuan paint weights in %s", target_dir)
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2.1",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "hunyuan3d-dit-v2-1/*",
            "hunyuan3d-paintpbr-v2-1/*",
        ],
    )
    return target_dir


def ensure_realesrgan(weights_root: Path) -> Path:
    target_dir = weights_root / "upscale"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "RealESRGAN_x4plus.pth"
    if not target_path.exists():
        raise FileNotFoundError(
            "RealESRGAN checkpoint not found. Place RealESRGAN_x4plus.pth in "
            f"{target_path} before running texture generation."
        )
    logger.info("Using RealESRGAN checkpoint at %s", target_path)
    return target_path


def decode_image(image_data_url: Optional[str]) -> Optional[Image.Image]:
    if not image_data_url:
        return None
    payload = image_data_url
    if "base64," in payload:
        _, payload = payload.split("base64,", 1)
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def decode_model(glb_base64: str, output_dir: Path) -> Path:
    raw = base64.b64decode(glb_base64)
    mesh = trimesh.load(io.BytesIO(raw), file_type="glb")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    obj_path = output_dir / "input_mesh.obj"
    mesh.export(str(obj_path))
    logger.info("Decoded GLB -> OBJ at %s", obj_path)
    return obj_path


def image_to_base64(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, image.width, image.height


def combine_metallic_roughness(metallic_path: Path, roughness_path: Path, output_path: Path) -> Path:
    metallic_img = Image.open(metallic_path).convert("L")
    roughness_img = Image.open(roughness_path).convert("L")

    if metallic_img.size != roughness_img.size:
        roughness_img = roughness_img.resize(metallic_img.size)

    width, height = metallic_img.size
    metallic_array = np.array(metallic_img)
    roughness_array = np.array(roughness_img)

    combined_array = np.zeros((height, width, 3), dtype=np.uint8)
    combined_array[:, :, 0] = 255
    combined_array[:, :, 1] = roughness_array
    combined_array[:, :, 2] = metallic_array

    Image.fromarray(combined_array).save(output_path)
    return output_path


def create_glb_with_pbr(obj_path: Path, textures: dict[str, Optional[Path]], output_path: Path) -> None:
    import pygltflib

    mesh = trimesh.load(obj_path)
    temp_glb = obj_path.parent / "temp_mesh.glb"
    mesh.export(temp_glb)

    gltf = pygltflib.GLTF2().load(temp_glb)

    def image_to_data_uri(image_path: Path) -> str:
        with image_path.open("rb") as handle:
            data = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    images: list[pygltflib.Image] = []
    textures_nodes: list[pygltflib.Texture] = []

    mapping = {
        "albedo": "baseColorTexture",
        "metallicRoughness": "metallicRoughnessTexture",
        "normal": "normalTexture",
        "ao": "occlusionTexture",
    }

    for kind, attribute in mapping.items():
        path = textures.get(kind)
        if not path:
            continue
        images.append(pygltflib.Image(uri=image_to_data_uri(path)))
        textures_nodes.append(pygltflib.Texture(source=len(images) - 1))

    pbr = pygltflib.PbrMetallicRoughness(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=1.0, roughnessFactor=1.0
    )

    texture_index = 0
    if textures.get("albedo"):
        pbr.baseColorTexture = pygltflib.TextureInfo(index=texture_index)
        texture_index += 1
    if textures.get("metallicRoughness"):
        pbr.metallicRoughnessTexture = pygltflib.TextureInfo(index=texture_index)
        texture_index += 1

    material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr)

    if textures.get("normal"):
        material.normalTexture = pygltflib.NormalTextureInfo(index=texture_index)
        texture_index += 1
    if textures.get("ao"):
        material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=texture_index)
        texture_index += 1

    gltf.images = images
    gltf.textures = textures_nodes
    gltf.materials = [material]

    if gltf.meshes:
        for primitive in gltf.meshes[0].primitives:
            primitive.material = 0

    gltf.save(str(output_path))
    temp_glb.unlink(missing_ok=True)


def patch_multiview_model(model: Any) -> None:
    if getattr(model, "_vf_patched", False):
        return

    original_forward_one = model.forward_one

    def forward_one_patched(input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        params = getattr(model, "_hy_params", None)
        seed = getattr(params, "seed", 0)
        view_size = custom_view_size or getattr(params, "view_resolution", model.pipeline.view_size)

        model.seed_everything(seed)

        if not isinstance(input_images, list):
            input_images_local = [input_images]
        else:
            input_images_local = input_images

        if not resize_input:
            resized_inputs = [img.resize((model.pipeline.view_size, model.pipeline.view_size)) for img in input_images_local]
        else:
            resized_inputs = [img.resize((view_size, view_size)) for img in input_images_local]

        processed_controls = []
        for ctrl in control_images:
            ctrl_resized = ctrl.resize((view_size, view_size))
            if ctrl_resized.mode == "L":
                ctrl_resized = ctrl_resized.point(lambda x: 255 if x > 1 else 0, mode="1")
            processed_controls.append(ctrl_resized)

        generator = torch.Generator(device=model.pipeline.device).manual_seed(seed)
        kwargs = dict(generator=generator)

        num_view = len(processed_controls) // 2
        normal_image = [[processed_controls[i] for i in range(num_view)]]
        position_image = [[processed_controls[i + num_view] for i in range(num_view)]]

        kwargs["width"] = view_size
        kwargs["height"] = view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(model.pipeline.unet, "use_dino") and model.pipeline.unet.use_dino:
            dino_hidden_states = model.dino_v2(resized_inputs[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        infer_steps = getattr(params, "num_inference_steps", None)
        guidance = getattr(params, "guidance_scale", 3.0)

        if infer_steps is None:
            infer_steps_dict = {
                "EulerAncestralDiscreteScheduler": 30,
                "UniPCMultistepScheduler": 15,
                "DDIMScheduler": 50,
                "ShiftSNRScheduler": 15,
            }
            infer_steps = infer_steps_dict.get(model.pipeline.scheduler.__class__.__name__, 30)

        images = model.pipeline(
            resized_inputs[0:1],
            num_inference_steps=infer_steps,
            prompt=prompt,
            sync_condition=None,
            guidance_scale=guidance,
            **kwargs,
        ).images

        if "pbr" in getattr(model, "mode", ""):
            images = {"albedo": images[:num_view], "mr": images[num_view:]}
        else:
            images = {"hdr": images}
        return images

    model.forward_one = forward_one_patched
    model._vf_patched = True


def configure_multiview_model(pipeline: Any, params: TextureWorkerParams) -> None:
    multiview_model = pipeline.models.get("multiview_model")
    if multiview_model is None:
        raise RuntimeError("Failed to initialize multiview diffusion model")

    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    multiview_model.seed_everything = seed_everything  # type: ignore
    multiview_model._hy_params = params
    patch_multiview_model(multiview_model)


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    weights_root = Path(payload["weights_root"])
    repo_root = Path(payload["repo_root"])

    configure_caches(weights_root)
    ensure_paths([str(repo_root), str(repo_root / "hy3dpaint"), str(repo_root / "hy3dpaint" / "hunyuanpaintpbr"), str(repo_root / "hy3dshape")])
    apply_torchvision_fix(repo_root)

    ensure_paint_weights(weights_root)
    realesrgan_path = ensure_realesrgan(weights_root)

    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline  # type: ignore
    from hy3dpaint.utils.pipeline_utils import ViewProcessor  # noqa: F401  # ensure module import for side effects

    params = TextureWorkerParams(
        seed=int(payload["seed"]),
        randomize_seed=bool(payload["randomize_seed"]),
        max_view_count=int(payload["max_view_count"]),
        view_resolution=int(payload["view_resolution"]),
        num_inference_steps=int(payload["num_inference_steps"]),
        guidance_scale=float(payload["guidance_scale"]),
        target_face_count=int(payload.get("target_face_count", 40000)),
        remesh_mesh=bool(payload["remesh_mesh"]),
        decimate=bool(payload.get("decimate", True)),
        uv_unwrap=bool(payload.get("uv_unwrap", True)),
        texture_resolution=int(payload.get("texture_resolution", 2048)),
        unload_model_after_generation=bool(payload["unload_model_after_generation"]),
    )

    final_seed = params.seed
    if params.randomize_seed:
        final_seed = random.randint(0, 2**32 - 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Hunyuan3DPaintConfig(max_num_view=params.max_view_count, resolution=params.view_resolution)
    config.target_face_count = params.target_face_count
    config.device = device.type
    config.render_size = params.texture_resolution
    config.texture_size = params.texture_resolution * 2
    config.multiview_cfg_path = str(repo_root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.custom_pipeline = "hunyuanpaintpbr"
    config.realesrgan_ckpt_path = str(realesrgan_path)

    logger.info(
        "Initialising paint pipeline (views=%d, resolution=%d, steps=%d, guidance=%.2f, target_faces=%d, decimate=%s, uv_unwrap=%s)",
        params.max_view_count,
        params.view_resolution,
        params.num_inference_steps,
        params.guidance_scale,
        params.target_face_count,
        params.decimate,
        params.uv_unwrap,
    )

    paint_pipeline = Hunyuan3DPaintPipeline(config)
    params.seed = final_seed
    configure_multiview_model(paint_pipeline, params)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        obj_path = decode_model(payload["model_base64"], tmpdir)

        reference_image = decode_image(payload.get("image_data_url"))
        if reference_image is None:
            raise RuntimeError("Texture generation requires an input reference image")

        output_obj_path = tmpdir / "textured_mesh.obj"
        logger.info(
            "Running paint pipeline (remesh=%s, target_faces=%d, decimate=%s, uv_unwrap=%s, view_res=%d, views=%d, steps=%d) ...",
            params.remesh_mesh,
            params.target_face_count,
            params.decimate,
            params.uv_unwrap,
            params.view_resolution,
            params.max_view_count,
            params.num_inference_steps,
        )

        try:
            paint_pipeline(
                mesh_path=str(obj_path),
                image_path=reference_image,
                output_mesh_path=str(output_obj_path),
                use_remesh=params.remesh_mesh,
                decimate=params.decimate,
                uv_unwrap=params.uv_unwrap,
                save_glb=False,
                target_face_count=params.target_face_count,
            )
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(
                "Paint pipeline out-of-memory. Consider lowering view resolution, view count, or target face count. %s",
                exc,
            )
            raise
        logger.info("Paint pipeline completed, textured OBJ at %s", output_obj_path)

        base_path = output_obj_path.with_suffix("")
        albedo_path = base_path.with_suffix(".jpg")
        metallic_path = base_path.with_name(base_path.name + "_metallic.jpg")
        roughness_path = base_path.with_name(base_path.name + "_roughness.jpg")

        if not albedo_path.exists():
            raise RuntimeError("Texture pipeline did not produce albedo texture")
        if not metallic_path.exists() or not roughness_path.exists():
            raise RuntimeError("Texture pipeline did not produce metallic/roughness textures")

        mr_combined_path = combine_metallic_roughness(
            metallic_path,
            roughness_path,
            base_path.with_name(base_path.name + "_mr.png"),
        )

        textured_glb_path = tmpdir / "textured_mesh.glb"
        create_glb_with_pbr(
            output_obj_path,
            {
                "albedo": albedo_path,
                "metallicRoughness": mr_combined_path,
            },
            textured_glb_path,
        )
        logger.info("Created textured GLB at %s", textured_glb_path)

        glb_bytes = textured_glb_path.read_bytes()
        albedo_base64, albedo_width, albedo_height = image_to_base64(albedo_path)
        mr_base64, mr_width, mr_height = image_to_base64(mr_combined_path)

    if params.unload_model_after_generation:
        del paint_pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "glb_base64": base64.b64encode(glb_bytes).decode("utf-8"),
        "file_name": "hunyuan-textured.glb",
        "mime_type": "model/gltf-binary",
        "seed": int(final_seed),
        "albedo_base64": albedo_base64,
        "albedo_file_name": "hunyuan-albedo.jpg",
        "albedo_mime_type": "image/jpeg",
        "albedo_width": albedo_width,
        "albedo_height": albedo_height,
        "rm_base64": mr_base64,
        "rm_file_name": "hunyuan-metallic-roughness.png",
        "rm_mime_type": "image/png",
        "rm_width": mr_width,
        "rm_height": mr_height,
    }


def main() -> None:
    raw_input = sys.stdin.read()
    payload = json.loads(raw_input)
    with contextlib.redirect_stdout(sys.stderr):
        result = run_worker(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
