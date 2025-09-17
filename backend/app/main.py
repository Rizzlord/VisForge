from __future__ import annotations

from collections import deque
import base64
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .triposg_service import TripoParams, triposg_service

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
