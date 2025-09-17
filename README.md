# VisForge

VisForge is a playground for building a ComfyUI-inspired visual node editor with an Unreal Engine aesthetic. The current milestone focuses on a rich React + Rete.js frontend, Babylon.js powered previews, and a FastAPI backend skeleton that can accept serialized graphs for future execution.

## Frontend (Rete.js + Babylon.js)

### Features
- Unreal‑style node styling with collapsible bodies and floating connection orbs.
- Node library sidebar grouped by category for quick drag-free placement.
- Workflow tabs with local persistence (save, save‑as, rename) backed by `localStorage`.
- Prebuilt nodes: **Load Image**, **Load Model**, **Separate Channels**, **Combine Channels**, **Show Image**, **Preview 3D**, **Generate Tripo Model**, **Save Model**, and **Save Image**.
- New **Generate Tripo Model** node that submits an image to the backend TripoSG pipeline and returns a GLB for downstream preview.
- Image pipeline helpers that split/merge RGBA channels directly in the browser.
- Babylon.js viewport with switchable **Base**, **Wire**, and **Norm** shading modes and smarter GLB/GLTF loading.

### Getting Started
```bash
cd frontend
npm install
npm run dev
```
The dev server runs on <http://localhost:5173>. The first load may take a few seconds because Babylon.js is bundled.

> Optionally create a `.env.local` in `frontend/` with `VITE_BACKEND_URL=http://localhost:8000` so the TripoSG node points at the FastAPI server during development.

## Backend (FastAPI)

A lightweight execution service is scaffolded at `backend/app/main.py`. It currently validates graphs and returns a topological execution plan, streaming placeholder updates over a WebSocket.

### Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

> **Note:** The TripoSG node relies on heavy ML dependencies (PyTorch, diffusers, pymeshlab, etc.) and the initial run downloads the required checkpoints into `backend/repos/TripoSG/weights/`. Ensure you have sufficient disk space and (optionally) a CUDA-capable GPU for reasonable performance.
The OpenAPI docs will be available at <http://localhost:8000/docs> once the server is running.

## Project Layout
```
VisForge/
├─ frontend/          # Vite + React + Rete.js + Babylon.js client
├─ backend/           # FastAPI service skeleton
└─ README.md          # This file
```

## Next Steps
1. Implement secure node execution inside the FastAPI service (e.g., subprocess sandbox or worker queue) and bridge it with the frontend WebSocket events.
2. Persist graphs and assets on the backend (e.g., PostgreSQL or SQLite + blob storage).
3. Add authentication/authorization before exposing execution endpoints publicly.
4. Extend node catalog (texture transforms, lighting, procedural nodes) and support undo/redo via Rete plugins.

Happy forging!
