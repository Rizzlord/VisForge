from __future__ import annotations

import base64
import contextlib
import gc
import io
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class UpscaleWorkerParams:
    unload_model_after_generation: bool


def ensure_paths(paths: list[str]) -> None:
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_caches(weights_root: Path) -> None:
    cache_dir = weights_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(weights_root / "torch"))


def ensure_realesrgan(weights_root: Path) -> Path:
    target_dir = weights_root
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "RealESRGAN_x4plus.pth"
    if not target_path.exists():
        raise FileNotFoundError(
            "RealESRGAN checkpoint not found. Place RealESRGAN_x4plus.pth in "
            f"{target_path} before running upscaling."
        )
    logger.info("Using RealESRGAN checkpoint at %s", target_path)
    return target_path


def decode_image(image_data_url: str) -> Image.Image:
    payload = image_data_url
    if "base64," in payload:
        _, payload = payload.split("base64,", 1)
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_to_base64(path: Path) -> tuple[str, int, int]:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, image.width, image.height


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    weights_root = Path(payload["weights_root"])
    repo_root = Path(payload["repo_root"])

    configure_caches(weights_root)
    # Add Hunyuan3D-2.1 to path to import realesrgan
    ensure_paths([str(repo_root.parent / "Hunyuan3D-2.1"), str(repo_root.parent / "Hunyuan3D-2.1" / "hy3dpaint"), str(repo_root.parent / "Hunyuan3D-2.1" / "hy3dpaint" / "hunyuanpaintpbr"), str(repo_root.parent / "Hunyuan3D-2.1" / "hy3dshape")])

    realesrgan_path = ensure_realesrgan(weights_root)

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    params = UpscaleWorkerParams(
        unload_model_after_generation=bool(payload["unload_model_after_generation"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(realesrgan_path),
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True, # need to check if this is ok
        gpu_id=None, # need to check if this is ok
    )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        input_image = decode_image(payload["image_data_url"])

        # convert to cv2 image
        cv2_image = np.array(input_image)
        cv2_image = cv2_image[:, :, ::-1].copy()

        try:
            output, _ = upsampler.enhance(cv2_image, outscale=4)
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(
                "Upscale pipeline out-of-memory. %s",
                exc,
            )
            raise
        logger.info("Upscale pipeline completed")

        # convert back to PIL image
        output = output[:, :, ::-1]
        output_image = Image.fromarray(output)

        output_path = tmpdir / "upscaled.png"
        output_image.save(output_path)

        image_base64, image_width, image_height = image_to_base64(output_path)

    if params.unload_model_after_generation:
        del upsampler
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "image_base64": image_base64,
        "file_name": "upscaled.png",
        "mime_type": "image/png",
        "width": image_width,
        "height": image_height,
    }


def main() -> None:
    raw_input = sys.stdin.read()
    payload = json.loads(raw_input)
    with contextlib.redirect_stdout(sys.stderr):
        result = run_worker(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
