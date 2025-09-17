# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

logger = logging.getLogger(__name__)


def remesh_mesh(mesh_path: str | Path, remesh_path: str | Path, target_count: int = 40000) -> None:
    logger.info("HyPaint remesh_mesh start: target_count=%d", target_count)
    mesh_simplify_trimesh(str(mesh_path), str(remesh_path), target_count=target_count)
    logger.info("HyPaint remesh_mesh finished: %s", remesh_path)


def mesh_simplify_trimesh(inputpath: str, outputpath: str, target_count: int = 40000) -> None:
    mesh = o3d.io.read_triangle_mesh(inputpath)
    if not mesh.has_triangles():
        raise ValueError("Input mesh has no faces")

    face_num = np.asarray(mesh.triangles).shape[0]
    logger.info("HyPaint simplify input faces=%d", face_num)

    target = max(int(target_count), 100)
    target = min(target, max(face_num - 1, 100))

    if face_num > target:
        logger.info("HyPaint applying Open3D quadric decimation to ~%d faces", target)
        mesh = mesh.simplify_quadric_decimation(target)
    else:
        logger.info("HyPaint decimation skipped (faces <= target)")

    # Convert to trimesh for consistent export
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    logger.info("HyPaint simplified faces=%d", len(triangles))

    simplified = trimesh.Trimesh(vertices=vertices, faces=triangles)
    simplified.export(outputpath)
