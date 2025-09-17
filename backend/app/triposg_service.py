from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "TripoSG"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
WEIGHTS_ROOT = REPO_ROOT / "weights"
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT = 60 * 45  # 45 minutes safety timeout


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
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _serialize_request(self, image_data_url: str, params: TripoParams) -> dict[str, object]:
        payload = asdict(params)
        payload['image_data_url'] = image_data_url
        payload['weights_root'] = str(WEIGHTS_ROOT)
        payload['repo_root'] = str(REPO_ROOT)
        payload['scripts_root'] = str(SCRIPTS_ROOT)
        return payload

    async def generate(self, image_data_url: str, params: TripoParams) -> bytes:
        request_payload = self._serialize_request(image_data_url, params)
        request_json = json.dumps(request_payload)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / 'tripo_worker.py'),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin and process.stdout

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=request_json.encode('utf-8')),
            timeout=PROCESS_TIMEOUT,
        )

        if process.returncode != 0:
            detail = stderr.decode('utf-8', errors='ignore') or 'Tripo worker failed'
            raise RuntimeError(detail)

        response = json.loads(stdout.decode('utf-8'))
        if 'glb_base64' not in response:
            raise RuntimeError('Worker response missing glb_base64 field')
        return base64.b64decode(response['glb_base64'])


triposg_service = TripoSGService()
