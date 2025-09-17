from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="VisForge Execution API", version="0.1.0")


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
