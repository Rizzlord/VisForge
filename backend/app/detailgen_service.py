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

        # Create unbuffered command for real-time output
        from .main import SubprocessLogger
        cmd = [python_executable, str(worker_path)]
        cmd = SubprocessLogger.create_unbuffered_process_args(cmd)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin and process.stdout

        # Send input and close stdin
        process.stdin.write(request_json.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        # Start streaming stderr in real-time
        stderr_task = asyncio.create_task(
            SubprocessLogger.log_stream(process.stderr, "DetailGen3D", "refinement")
        )

        # Read stdout and wait for process completion
        try:
            stdout = await asyncio.wait_for(process.stdout.read(), timeout=PROCESS_TIMEOUT)
            await asyncio.wait_for(process.wait(), timeout=1)
        finally:
            await stderr_task

        if process.returncode != 0:
            SubprocessLogger.log_error("DetailGen3D", "refinement", f"Worker failed with return code {process.returncode}")
            raise RuntimeError(f"DetailGen3D worker failed with return code {process.returncode}")

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
