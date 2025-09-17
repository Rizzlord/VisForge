from __future__ import annotations

import asyncio
import logging
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "RMBG-2.0"
WEIGHTS_PATH = REPO_ROOT / "rmbg" / "model.safetensors"


@lru_cache(maxsize=1)
def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


class RMBGService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._model: Optional[torch.nn.Module] = None
        self._model_lock = Lock()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self._transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            if not WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"RMBG weights not found at {WEIGHTS_PATH}")

            _ensure_repo_on_path()
            from BiRefNet_config import BiRefNetConfig  # type: ignore
            from birefnet import BiRefNet  # type: ignore
            from safetensors.torch import load_file  # type: ignore

            config = BiRefNetConfig(bb_pretrained=False)
            model = BiRefNet(bb_pretrained=False, config=config)

            state_dict = load_file(str(WEIGHTS_PATH))
            # Handle common prefixes such as 'module.' from DataParallel checkpoints
            cleaned_state_dict = {self._strip_prefix(k): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
            if missing:
                logger.warning("RMBG model missing expected keys: %s", missing)
            if unexpected:
                logger.warning("RMBG model encountered unexpected keys: %s", unexpected)

            model.eval()
            model.to(self._device, dtype=self._dtype)

            self._model = model
            return model

    @staticmethod
    def _strip_prefix(key: str) -> str:
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def remove_background(self, image: Image.Image) -> Image.Image:
        model = self._load_model()

        original_size = image.size  # (width, height)
        rgb_image = image.convert("RGB")
        tensor = self._transform(rgb_image).unsqueeze(0)
        tensor = tensor.to(self._device, dtype=self._dtype)

        with torch.no_grad():
            outputs = model(tensor)

        logits = outputs
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        if logits.ndim == 3:
            logits = logits.unsqueeze(0)

        mask = torch.sigmoid(logits)
        mask = F.interpolate(
            mask,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        mask = mask[0, 0].clamp_(0, 1)

        alpha = (mask * 255).to(torch.uint8).cpu().numpy()
        alpha_image = Image.fromarray(alpha, mode="L")

        rgba_image = rgb_image.convert("RGBA")
        rgba_image.putalpha(alpha_image)
        return rgba_image


rmbg_service = RMBGService()
