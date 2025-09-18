from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from rmbg_service import rmbg_service  # noqa: E402


def _decode_image(data_url: str) -> Image.Image:
    payload = data_url
    if 'base64,' in payload:
        _, payload = payload.split('base64,', 1)
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes))


def _parse_hex_color(value: str | None) -> Tuple[int, int, int]:
    if not value:
        return (255, 255, 255)
    hex_value = value.lstrip('#')
    return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))


def _composite_to_rgb(image: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    background = Image.new('RGB', image.size, color)
    alpha = image.split()[3]
    background.paste(image.convert('RGB'), mask=alpha)
    return background


def _encode_image(image: Image.Image) -> Tuple[str, int, int]:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded, image.width, image.height


def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    pil_image = _decode_image(payload['image_data_url'])
    removed_image = rmbg_service.remove_background(
        pil_image,
        unload_model=bool(payload.get('unload_model', True)),
    )

    transparent = bool(payload.get('transparent', True))
    mode = payload.get('mode', 'rgba')

    if transparent:
        output_image = removed_image if mode == 'rgba' else removed_image.convert('RGBA')
    else:
        color = _parse_hex_color(payload.get('color'))
        flattened = _composite_to_rgb(removed_image, color)
        output_image = flattened if mode == 'rgb' else flattened.convert('RGBA')

    image_base64, width, height = _encode_image(output_image)

    return {
        'image_base64': image_base64,
        'file_name': payload.get('file_name', 'removed-background.png'),
        'mime_type': 'image/png',
        'width': width,
        'height': height,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    result = process(payload)
    json.dump(result, sys.stdout)


if __name__ == '__main__':
    main()
