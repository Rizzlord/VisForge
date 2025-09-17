from __future__ import annotations

from typing import Optional

from PIL import Image


def _ensure_rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def flatten_to_white(rgba: Image.Image) -> Image.Image:
    """Return an RGB image composited onto a white background."""

    rgba_image = _ensure_rgba(rgba)
    alpha = rgba_image.split()[3] if rgba_image.mode == "RGBA" else None

    base = Image.new("RGB", rgba_image.size, (255, 255, 255))
    if alpha is not None:
        base.paste(rgba_image.convert("RGB"), mask=alpha)
    else:
        base.paste(rgba_image.convert("RGB"))
    return base


def prepare_input_image(
    image: Image.Image,
    *,
    remover: Optional[object] = None,
    remove_background: bool,
    apply_preparation: bool,
) -> Image.Image:
    """Apply repo-consistent image prep for Hunyuan3D."""

    if not apply_preparation:
        if remove_background and remover is not None:
            processed = remover(image.convert("RGB"))
            return _ensure_rgba(processed)
        return _ensure_rgba(image)

    working = image
    if remove_background and remover is not None:
        working = remover(image.convert("RGB"))

    return flatten_to_white(working)
