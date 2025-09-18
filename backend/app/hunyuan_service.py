from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "Hunyuan3D-2.1"
VENV_DIR: Final[Path] = REPO_ROOT / "venv"
WEIGHTS_ROOT = REPO_ROOT / "weights"
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT = 60 * 45  # 45 minute safety timeout


@dataclass
class HunyuanParams:
    seed: int
    randomize_seed: bool
    remove_background: bool
    num_inference_steps: int
    guidance_scale: float
    octree_resolution: int
    num_chunks: int
    mc_algo: str
    unload_model_after_generation: bool
    use_repo_venv: bool = False


class HunyuanService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _serialize_request(self, image_data_url: str, params: HunyuanParams) -> dict[str, object]:
        payload = asdict(params)
        payload["image_data_url"] = image_data_url
        payload["weights_root"] = str(WEIGHTS_ROOT)
        payload["repo_root"] = str(REPO_ROOT)
        return payload

    async def generate(self, image_data_url: str, params: HunyuanParams) -> tuple[bytes, dict[str, object]]:
        request_payload = self._serialize_request(image_data_url, params)
        request_json = json.dumps(request_payload)

        python_executable = self._resolve_python(params.use_repo_venv)

        cmd = [
            python_executable,
            str(Path(__file__).resolve().parent / "hunyuan_worker.py"),
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
            detail = stderr.decode("utf-8", errors="ignore") or "Hunyuan worker failed"
            raise RuntimeError(detail)

        response = json.loads(stdout.decode("utf-8"))
        if "glb_base64" not in response:
            raise RuntimeError("Worker response missing glb_base64 field")

        glb_bytes = base64.b64decode(response["glb_base64"])
        return glb_bytes, response

    def _resolve_python(self, use_repo_venv: bool) -> str:
        if not use_repo_venv:
            return sys.executable

        if sys.platform.startswith('win'):
            candidate = VENV_DIR / 'Scripts' / 'python.exe'
        else:
            candidate = VENV_DIR / 'bin' / 'python'

        if not candidate.exists():
            raise RuntimeError(f"Virtual environment python executable not found at {candidate}")

        return str(candidate)


hunyuan_service = HunyuanService()
