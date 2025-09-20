from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "DetailGen3D"
VENV_DIR: Final[Path] = REPO_ROOT / "venv"
WEIGHTS_ROOT = REPO_ROOT / "pretrained_weights"
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT = 60 * 60  # 60 minute safety timeout


@dataclass
class DetailGen3DParams:
    seed: int
    num_inference_steps: int
    guidance_scale: float
    noise_aug: float
    use_repo_venv: bool = False
    unload_model_after_generation: bool = True


class DetailGen3DService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def _serialize_request(self, model_base64: str, image_data_url: str, params: DetailGen3DParams) -> dict[str, object]:
        payload = asdict(params)
        payload["model_base64"] = model_base64
        payload["image_data_url"] = image_data_url
        payload["repo_root"] = str(REPO_ROOT)
        payload["weights_root"] = str(WEIGHTS_ROOT)
        return payload

    async def refine(self, model_base64: str, image_data_url: str, params: DetailGen3DParams) -> bytes:
        request_payload = self._serialize_request(model_base64, image_data_url, params)
        request_json = json.dumps(request_payload)

        python_executable = self._resolve_python(params.use_repo_venv)
        worker_path = Path(__file__).resolve().parent / "detailgen_worker.py"

        process = await asyncio.create_subprocess_exec(
            python_executable,
            str(worker_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin and process.stdout

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=request_json.encode("utf-8")),
            timeout=PROCESS_TIMEOUT,
        )

        # Log stderr output to Python logs
        from .main import add_python_log
        if stderr:
            stderr_text = stderr.decode("utf-8", errors="ignore")
            for line in stderr_text.split('\n'):
                if line.strip():
                    add_python_log("INFO", f"[detailgen] {line.strip()}", "detailgen3d")

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore") or "DetailGen3D worker failed"
            add_python_log("ERROR", f"[detailgen] Worker failed: {detail}", "detailgen3d")
            raise RuntimeError(detail)

        response = json.loads(stdout.decode("utf-8"))
        if "glb_base64" not in response:
            raise RuntimeError("Worker response missing glb_base64 field")

        return base64.b64decode(response["glb_base64"])

    def _resolve_python(self, use_repo_venv: bool) -> str:
        if not use_repo_venv:
            return sys.executable

        if sys.platform.startswith("win"):
            candidate = VENV_DIR / "Scripts" / "python.exe"
        else:
            candidate = VENV_DIR / "bin" / "python"

        if not candidate.exists():
            raise RuntimeError(f"Virtual environment python executable not found at {candidate}")

        return str(candidate)


detailgen_service = DetailGen3DService()
