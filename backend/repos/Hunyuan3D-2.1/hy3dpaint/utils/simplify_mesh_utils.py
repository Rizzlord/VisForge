# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import trimesh
import pymeshlab

logger = logging.getLogger(__name__)


def remesh_mesh(mesh_path, remesh_path, target_count=40000):
    logger.info("HyPaint remesh_mesh start: target_count=%d", target_count)
    mesh_simplify_trimesh(mesh_path, remesh_path, target_count=target_count)
    logger.info("HyPaint remesh_mesh finished: %s", remesh_path)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # 先去除离散面
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    # 调用减面函数
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]

    if face_num > target_count:
        reduction = target_count / face_num
        logger.info(
            "HyPaint simplify mesh: faces=%d target=%d reduction=%.4f",
            face_num,
            target_count,
            reduction,
        )
        try:
            courent = courent.simplify_quadric_decimation(target_count=target_count)
        except Exception as exc:
            logger.warning("HyPaint simplify_quadric_decimation failed: %s", exc)
    courent.export(outputpath)
    logger.info("HyPaint mesh_simplify_trimesh exported %s", outputpath)
