from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "Hunyuan3D-2.1"
WEIGHTS_ROOT = REPO_ROOT / "weights"
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT = 60 * 60  # 60 minute safety timeout


@dataclass
class HunyuanTextureParams:
    seed: int
    randomize_seed: bool
    max_view_count: int
    view_resolution: int
    num_inference_steps: int
    guidance_scale: float
    target_face_count: int
    remesh_mesh: bool
    unload_model_after_generation: bool


class HunyuanTextureService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _serialize_request(
        self,
        model_base64: str,
        image_data_url: str,
        params: HunyuanTextureParams,
    ) -> Dict[str, Any]:
        payload = asdict(params)
        payload["model_base64"] = model_base64
        payload["image_data_url"] = image_data_url
        payload["weights_root"] = str(WEIGHTS_ROOT)
        payload["repo_root"] = str(REPO_ROOT)
        return payload

    async def generate(
        self,
        model_base64: str,
        image_data_url: str,
        params: HunyuanTextureParams,
    ) -> Dict[str, Any]:
        request_payload = self._serialize_request(model_base64, image_data_url, params)
        request_json = json.dumps(request_payload)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "hunyuan_texture_worker.py"),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin and process.stdout

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=request_json.encode("utf-8")),
            timeout=PROCESS_TIMEOUT,
        )

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore") or "Hunyuan texture worker failed"
            raise RuntimeError(detail)

        response = json.loads(stdout.decode("utf-8"))
        required_keys = {"glb_base64", "albedo_base64", "rm_base64"}
        if not required_keys.issubset(response.keys()):
            raise RuntimeError("Texture worker response missing required fields")

        return response


hunyuan_texture_service = HunyuanTextureService()
