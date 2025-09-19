Setup instructions for Upscale worker

This folder contains the Upscale worker and model weights. The worker depends on several Python packages which are heavy (PyTorch, Real-ESRGAN, BasicSR). Two ways to enable the worker:

Option A — create a repository virtual environment (recommended)

1. From project root run:

```bash
python3 -m venv backend/repos/Upscale/venv
source backend/repos/Upscale/venv/bin/activate
pip install --upgrade pip
# Install PyTorch using the recommended wheel for your platform; example CPU-only:
pip install torch -f https://download.pytorch.org/whl/torch_stable.html
# Then install other dependencies
pip install -r backend/repos/Upscale/requirements.txt
```

2. After venv creation, set the Upscale node control to `Use repo venv` in the UI or enable `use_repo_venv` when sending requests.

Option B — install dependencies in system Python (not recommended)

```bash
pip install -r backend/repos/Upscale/requirements.txt
# plus install torch as appropriate for your system
```

Notes
- The RealESRGAN checkpoint file `RealESRGAN_x4plus.pth` must be present in this directory; the repository already includes it.
- Installing PyTorch and RealESRGAN may take a long time and require significant disk space. Use the appropriate PyTorch wheel for your CUDA or CPU setup.
- If you want, I can attempt to create the venv and install the packages here, but that requires network access and can be slow. Let me know if you want me to proceed.
