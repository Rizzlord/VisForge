from __future__ import annotations

from collections import deque
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import networkx as nx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
from starlette.concurrency import run_in_threadpool

from .triposg_service import TripoParams, triposg_service
from .hunyuan_service import HunyuanParams, hunyuan_service
from .rmbg_service import rmbg_service
from PIL import Image

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = BACKEND_ROOT / "workflows"
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
WORKFLOWS_INDEX_FILE = WORKFLOWS_DIR / "index.json"

app = FastAPI(title="VisForge Execution API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index() -> JSONResponse:
    """Provide a friendly landing response for the root path."""

    return JSONResponse(
        {
            "status": "ok",
            "message": "VisForge backend is running.",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }
    )


class NodePayload(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type label")
    params: Dict[str, Any] = Field(default_factory=dict)


class ConnectionPayload(BaseModel):
    id: Optional[str] = Field(default=None, description="Connection id if available")
    source: str = Field(..., description="Upstream node id")
    source_output: str = Field(..., description="Output key on the source node")
    target: str = Field(..., description="Downstream node id")
    target_input: str = Field(..., description="Input key on the target node")


class GraphPayload(BaseModel):
    nodes: List[NodePayload] = Field(default_factory=list)
    connections: List[ConnectionPayload] = Field(default_factory=list)


class WorkflowDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    active_workflow_id: Optional[str] = Field(default=None, alias="activeWorkflowId")
    workflows: List[WorkflowDocument] = Field(default_factory=list)


def _slugify_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "workflow"


def _reserve_filename(name: str, used: set[str]) -> str:
    base = _slugify_name(name)
    candidate = f"{base}.json"
    counter = 2
    while candidate in used:
        candidate = f"{base}-{counter}.json"
        counter += 1
    used.add(candidate)
    return candidate


def _load_workflow_state() -> WorkflowState:
    if not WORKFLOWS_INDEX_FILE.exists():
        return WorkflowState(active_workflow_id=None, workflows=[])

    try:
        with WORKFLOWS_INDEX_FILE.open("r", encoding="utf-8") as index_file:
            index_payload = json.load(index_file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read workflow index: %s", exc)
        return WorkflowState(active_workflow_id=None, workflows=[])

    workflows: List[WorkflowDocument] = []

    for entry in index_payload.get("workflows", []):
        file_name = entry.get("file")
        if not file_name:
            continue

        file_path = WORKFLOWS_DIR / file_name
        if not file_path.exists():
            logger.warning("Workflow file missing: %s", file_name)
            continue

        try:
            with file_path.open("r", encoding="utf-8") as workflow_file:
                payload = json.load(workflow_file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load workflow '%s': %s", file_name, exc)
            continue

        workflow_id = str(payload.get("id") or entry.get("id") or "").strip()
        if not workflow_id:
            logger.warning("Workflow file '%s' missing id", file_name)
            continue

        workflow_name = str(payload.get("name") or entry.get("name") or "Untitled Workflow").strip()
        workflow_data = payload.get("data")
        if not isinstance(workflow_data, dict):
            workflow_data = {"nodes": [], "connections": []}

        workflows.append(
            WorkflowDocument(id=workflow_id, name=workflow_name, data=workflow_data)
        )

    active_workflow_id = index_payload.get("activeWorkflowId")
    if isinstance(active_workflow_id, str):
        active_workflow_id = active_workflow_id.strip() or None
    else:
        active_workflow_id = None

    return WorkflowState(active_workflow_id=active_workflow_id, workflows=workflows)


def _save_workflow_state(state: WorkflowState) -> WorkflowState:
    used_file_names: set[str] = set()
    persisted_entries: List[Dict[str, str]] = []

    for document in state.workflows:
        file_name = _reserve_filename(document.name, used_file_names)
        file_path = WORKFLOWS_DIR / file_name

        payload = {
            "id": document.id,
            "name": document.name,
            "data": document.data,
        }

        try:
            with file_path.open("w", encoding="utf-8") as workflow_file:
                json.dump(payload, workflow_file, indent=2)
        except OSError as exc:
            logger.error("Failed to persist workflow '%s': %s", document.name, exc)
            raise

        persisted_entries.append({"id": document.id, "name": document.name, "file": file_name})

    retained_files = {entry["file"] for entry in persisted_entries}

    for path in WORKFLOWS_DIR.glob("*.json"):
        if path.name == WORKFLOWS_INDEX_FILE.name:
            continue
        if path.name not in retained_files:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove stale workflow file '%s': %s", path.name, exc)

    index_payload = {
        "activeWorkflowId": state.active_workflow_id,
        "workflows": persisted_entries,
    }

    try:
        with WORKFLOWS_INDEX_FILE.open("w", encoding="utf-8") as index_file:
            json.dump(index_payload, index_file, indent=2)
    except OSError as exc:
        logger.error("Failed to write workflow index: %s", exc)
        raise

    return state


@app.get("/workflows/state", response_model=WorkflowState)
async def get_workflow_state_endpoint() -> WorkflowState:
    return await run_in_threadpool(_load_workflow_state)


@app.put("/workflows/state", response_model=WorkflowState)
async def put_workflow_state_endpoint(state: WorkflowState) -> WorkflowState:
    validated = WorkflowState.model_validate(state)
    return await run_in_threadpool(_save_workflow_state, validated)


class TripoSGRequest(BaseModel):
    image_data_url: str = Field(..., description="Image data URL (base64) to convert into a 3D model")
    seed: int = Field(681589206, ge=0, le=0xFFFFFFFFFFFFFFFF)
    use_float16: bool = True
    extra_depth_level: int = Field(1, ge=0, le=4)
    num_inference_steps: int = Field(50, ge=1, le=1000)
    cfg_scale: float = Field(15.0, ge=0.0, le=50.0)
    simplify_mesh: bool = False
    target_face_number: int = Field(100000, ge=500, le=10_000_000)
    use_flash_decoder: bool = True
    dense_octree_resolution: int = Field(512, description="Dense octree resolution (power of two)")
    hierarchical_octree_resolution: int = Field(512, description="Hierarchical octree base resolution (power of two)")
    flash_octree_resolution: int = Field(512, description="Flash octree resolution (power of two)")
    unload_model_after_generation: bool = True

    @model_validator(mode="after")
    def _validate_resolutions(self) -> 'TripoSGRequest':
        for attr in (
            "dense_octree_resolution",
            "hierarchical_octree_resolution",
            "flash_octree_resolution",
        ):
            value = getattr(self, attr)
            if value <= 0 or (value & (value - 1)) != 0:
                raise ValueError(f"{attr} must be a power-of-two positive integer")
        return self


class TripoSGResponse(BaseModel):
    glb_base64: str
    file_name: str = "tripo-model.glb"
    mime_type: str = "model/gltf-binary"


class HunyuanRequest(BaseModel):
    image_data_url: str = Field(..., description="Image data URL (base64) to convert into a 3D model")
    seed: int = Field(1234, ge=0, le=0xFFFFFFFF)
    randomize_seed: bool = Field(False, description="Randomize seed before generation")
    remove_background: bool = Field(True, description="Apply background removal before inference")
    num_inference_steps: int = Field(30, ge=1, le=200)
    guidance_scale: float = Field(5.0, ge=0.0, le=50.0)
    octree_resolution: int = Field(256, ge=16, le=1024)
    num_chunks: int = Field(8000, ge=1000, le=5_000_000)
    mc_algo: Literal['mc', 'dmc'] = Field('dmc', description="Surface extraction algorithm")
    unload_model_after_generation: bool = Field(True, description="Release model weights after generation")


class HunyuanResponse(BaseModel):
    glb_base64: str
    file_name: str = "hunyuan-model.glb"
    mime_type: str = "model/gltf-binary"
    seed: int


@app.post("/execute")
async def execute_graph(graph: GraphPayload) -> Dict[str, Any]:
    """Validate and topologically order the submitted graph."""

    dag = nx.DiGraph()
    for node in graph.nodes:
        dag.add_node(node.id, type=node.type, params=node.params)

    for connection in graph.connections:
        dag.add_edge(connection.source, connection.target, key=connection.source_output)

    if not nx.is_directed_acyclic_graph(dag):
        return {
            "status": "rejected",
            "reason": "Graph contains at least one cycle.",
        }

    ordered_nodes = list(nx.topological_sort(dag))

    plan: List[Dict[str, Any]] = []
    for node_id in ordered_nodes:
        upstream = [edge[0] for edge in dag.in_edges(node_id)]
        plan.append({
            "id": node_id,
            "type": dag.nodes[node_id]["type"],
            "depends_on": upstream,
        })

    return {
        "status": "accepted",
        "steps": plan,
    }


@app.websocket("/ws/execution")
async def execution_progress(websocket: WebSocket) -> None:
    """Streams placeholder execution events to connected clients."""

    await websocket.accept()
    phases = deque([
        {"phase": "queued", "message": "Awaiting worker slot."},
        {"phase": "running", "message": "Execution started."},
        {"phase": "completed", "message": "Graph finished."},
    ])

    try:
        while phases:
            await websocket.send_json(phases.popleft())
    except WebSocketDisconnect:
        return


@app.post("/triposg/generate", response_model=TripoSGResponse)
async def triposg_generate(request: TripoSGRequest) -> TripoSGResponse:
    params = TripoParams(
        seed=request.seed,
        use_float16=request.use_float16,
        extra_depth_level=request.extra_depth_level,
        num_inference_steps=request.num_inference_steps,
        cfg_scale=request.cfg_scale,
        simplify_mesh=request.simplify_mesh,
        target_face_number=request.target_face_number,
        use_flash_decoder=request.use_flash_decoder,
        dense_octree_resolution=request.dense_octree_resolution,
        hierarchical_octree_resolution=request.hierarchical_octree_resolution,
        flash_octree_resolution=request.flash_octree_resolution,
        unload_model_after_generation=request.unload_model_after_generation,
    )

    async with triposg_service.lock:
        try:
            glb_bytes = await triposg_service.generate(
                request.image_data_url,
                params,
            )
        except Exception as exc:  # pragma: no cover - runtime error surface
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
    return TripoSGResponse(glb_base64=glb_base64)


@app.post("/hunyuan/generate", response_model=HunyuanResponse)
async def hunyuan_generate(request: HunyuanRequest) -> HunyuanResponse:
    params = HunyuanParams(
        seed=request.seed,
        randomize_seed=request.randomize_seed,
        remove_background=request.remove_background,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        octree_resolution=request.octree_resolution,
        num_chunks=request.num_chunks,
        mc_algo=request.mc_algo,
        unload_model_after_generation=request.unload_model_after_generation,
    )

    async with hunyuan_service.lock:
        try:
            glb_bytes, metadata = await hunyuan_service.generate(
                request.image_data_url,
                params,
            )
        except Exception as exc:  # pragma: no cover - runtime error surface
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
    return HunyuanResponse(
        glb_base64=glb_base64,
        file_name=str(metadata.get("file_name", "hunyuan-model.glb")),
        mime_type=str(metadata.get("mime_type", "model/gltf-binary")),
        seed=int(metadata.get("seed", request.seed)),
    )


class RemoveBackgroundRequest(BaseModel):
    image_data_url: str = Field(..., description="Image data URL (base64) to process")
    mode: Literal['rgb', 'rgba'] = Field('rgba', description="Output mode")
    transparent: bool = Field(True, description="Return transparent background (ignored for RGB)")
    color: Optional[str] = Field('#ffffff', description="Hex color for non-transparent background")

    @model_validator(mode="after")
    def _validate_color(self) -> 'RemoveBackgroundRequest':
        if not self.transparent:
            if not self.color or not isinstance(self.color, str) or not self.color.startswith('#') or len(self.color) != 7:
                raise ValueError("Provide a color in #RRGGBB format when transparent is disabled")
        return self


class RemoveBackgroundResponse(BaseModel):
    image_base64: str
    file_name: str = "removed-background.png"
    mime_type: str = "image/png"
    width: int
    height: int


def _decode_image(data_url: str) -> Image.Image:
    payload = data_url
    if 'base64,' in payload:
        _, payload = payload.split('base64,', 1)
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes))


def _composite_to_rgb(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    background = Image.new('RGB', image.size, color)
    alpha = image.split()[3]
    background.paste(image.convert('RGB'), mask=alpha)
    return background


def _parse_hex_color(value: Optional[str]) -> tuple[int, int, int]:
    if not value:
        return (255, 255, 255)
    hex_value = value.lstrip('#')
    return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))


def _process_background(request: RemoveBackgroundRequest) -> RemoveBackgroundResponse:
    pil_image = _decode_image(request.image_data_url)
    removed_image = rmbg_service.remove_background(pil_image)

    if request.transparent:
        output_image = removed_image
        if request.mode == 'rgb':
            output_image = removed_image  # transparency implies RGBA output
    else:
        color = _parse_hex_color(request.color)
        flattened = _composite_to_rgb(removed_image, color)
        if request.mode == 'rgb':
            output_image = flattened
        else:
            output_image = flattened.convert('RGBA')

    buffer = io.BytesIO()
    output_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return RemoveBackgroundResponse(
        image_base64=encoded,
        width=output_image.width,
        height=output_image.height,
    )


@app.post("/image/remove_background", response_model=RemoveBackgroundResponse)
async def remove_background_endpoint(request: RemoveBackgroundRequest) -> RemoveBackgroundResponse:
    try:
        return await run_in_threadpool(_process_background, request)
    except Exception as exc:  # pragma: no cover - runtime error surface
        raise HTTPException(status_code=500, detail=str(exc)) from exc
