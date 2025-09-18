import os
import sys
import torch
import trimesh
import numpy as np
from PIL import Image
from skimage import measure
from huggingface_hub import snapshot_download

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NODE_DIR)

from detailgen3d.inference_utils import generate_dense_grid_points
from detailgen3d.pipelines.pipeline_detailgen3d import (
    DetailGen3DPipeline,
)

class DetailGen3DNode:
    _pipeline = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "noise_aug": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "refine"
    CATEGORY = "DetailGen3D"

    def _load_pipeline(self, device, dtype):
        if DetailGen3DNode._pipeline is None:
            print("Loading DetailGen3D pipeline...")
            local_dir = os.path.join(NODE_DIR, "pretrained_weights", "DetailGen3D")
            os.makedirs(local_dir, exist_ok=True)
            
            if not os.path.exists(os.path.join(local_dir, "model_index.json")):
                 print(f"Downloading DetailGen3D model to {local_dir}...")
                 snapshot_download(repo_id="VAST-AI/DetailGen3D", local_dir=local_dir)
            else:
                 print("DetailGen3D model found locally.")

            DetailGen3DNode._pipeline = DetailGen3DPipeline.from_pretrained(local_dir)
            print("DetailGen3D pipeline loaded.")
        
        return DetailGen3DNode._pipeline.to(device, dtype=dtype)

    def _preprocess_mesh(self, input_mesh, num_pc=20480):
        mesh = input_mesh.copy()

        center = mesh.bounding_box.centroid
        mesh.apply_translation(-center)
        scale = max(mesh.bounding_box.extents)
        if scale > 1e-6:
            mesh.apply_scale(1.9 / scale)

        try:
            surface, face_indices = trimesh.sample.sample_surface(mesh, 1000000)
            normal = mesh.face_normals[face_indices]
        except Exception as e:
            raise RuntimeError(f"Failed to sample surface from mesh. It might be invalid or have no faces. Error: {e}")

        if len(surface) == 0:
            raise RuntimeError("Mesh has no surface area to sample from.")

        num_pc_to_sample = min(num_pc, surface.shape[0])
        
        rng = np.random.default_rng()
        ind = rng.choice(surface.shape[0], num_pc_to_sample, replace=False)
        surface_tensor = torch.FloatTensor(surface[ind])
        normal_tensor = torch.FloatTensor(normal[ind])
        surface_tensor = torch.cat([surface_tensor, normal_tensor], dim=-1).unsqueeze(0)

        return surface_tensor

    def refine(self, mesh, image, seed, steps, guidance_scale, noise_aug):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipeline = self._load_pipeline(device, dtype)

        pil_image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")

        surface = self._preprocess_mesh(mesh).to(device, dtype=dtype)
        
        batch_size = 1

        box_min = np.array([-1.005, -1.005, -1.005])
        box_max = np.array([1.005, 1.005, 1.005])
        sampled_points, grid_size, bbox_size = generate_dense_grid_points(
            bbox_min=box_min, bbox_max=box_max, octree_depth=9, indexing="ij"
        )
        sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=dtype)
        sampled_points = sampled_points.unsqueeze(0).repeat(batch_size, 1, 1)

        print("Starting DetailGen3D inference...")
        sample = pipeline.vae.encode(surface).latent_dist.sample()
        sdf = pipeline(
            pil_image, 
            latents=sample, 
            sampled_points=sampled_points, 
            noise_aug_level=noise_aug, 
            generator=torch.Generator(device=device).manual_seed(seed),
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).samples[0]
        print("Inference complete.")

        print("Running Marching Cubes...")
        grid_logits = sdf.view(grid_size).cpu().numpy()
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logits, 0, method="lewiner"
        )
        vertices = vertices / grid_size * bbox_size + box_min
        
        output_mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
        print("Marching Cubes complete. Output mesh generated.")

        return (output_mesh,)