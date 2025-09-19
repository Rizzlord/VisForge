from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Final

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT / "repos" / "Upscale"
VENV_DIR: Final[Path] = REPO_ROOT / "venv"
WEIGHTS_ROOT = REPO_ROOT 
WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

PROCESS_TIMEOUT = 60 * 60  # 60 minute safety timeout

logger = logging.getLogger(__name__)


@dataclass
class UpscaleParams:
    unload_model_after_generation: bool
    use_repo_venv: bool = False


class UpscaleService:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def _log_stream(self, stream: asyncio.StreamReader, prefix: str) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info("[%s] %s", prefix, line.decode("utf-8", errors="ignore").rstrip())

    def _serialize_request(
        self,
        image_data_url: str,
        params: UpscaleParams,
    ) -> Dict[str, Any]:
        payload = asdict(params)
        payload["image_data_url"] = image_data_url
        payload["weights_root"] = str(WEIGHTS_ROOT)
        payload["repo_root"] = str(REPO_ROOT)
        return payload

    async def generate(
        self,
        image_data_url: str,
        params: UpscaleParams,
    ) -> Dict[str, Any]:
        request_payload = self._serialize_request(image_data_url, params)
        request_json = json.dumps(request_payload)

        python_executable = self._resolve_python(params.use_repo_venv)

        cmd = [
            python_executable,
            str(Path(__file__).resolve().parent / "upscale_worker.py"),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin and process.stdout and process.stderr

        process.stdin.write(request_json.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        stderr_task = asyncio.create_task(self._log_stream(process.stderr, "upscale"))

        try:
            stdout = await asyncio.wait_for(process.stdout.read(), timeout=PROCESS_TIMEOUT)
            await asyncio.wait_for(process.wait(), timeout=1)
        finally:
            await stderr_task

        if process.returncode != 0:
            detail = "Upscale worker failed"
            raise RuntimeError(detail)

        response = json.loads(stdout.decode("utf-8"))
        required_keys = {"image_base64"}
        if not required_keys.issubset(response.keys()):
            raise RuntimeError("Upscale worker response missing required fields")

        return response

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


upscale_service = UpscaleService()
