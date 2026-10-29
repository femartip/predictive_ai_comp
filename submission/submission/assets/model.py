"""Processors for supported 3D model MIME types."""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np
from PIL import Image

from .base import AssetProcessingContext, ProcessedAsset, description_result, evenly_spaced_indices


VIEWS = [
    ("front", 0, -90),
    ("back", 0, 90),
    ("left", 0, 180),
    ("right", 0, 0),
    ("top", 90, -90),
    ("isometric", 30, -45),
]


def _load_obj(data: bytes):
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("OBJ processing requires trimesh. Run `poetry install`.") from error
    loaded = trimesh.load(BytesIO(data), file_type="obj", force="scene", process=False)
    geometries = list(loaded.geometry.values())
    if not geometries:
        raise ValueError("OBJ contains no geometry")
    transformed = loaded.dump(concatenate=False)
    meshes = [geometry for geometry in transformed if isinstance(geometry, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("OBJ contains no mesh geometry")
    return trimesh.util.concatenate(meshes), geometries


def _geometry_summary(mesh, geometries) -> dict:
    materials = []
    for geometry in geometries:
        material = getattr(getattr(geometry, "visual", None), "material", None)
        name = getattr(material, "name", None)
        if name:
            materials.append(str(name))
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = None
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_components": components,
        "bounding_box_dimensions": [float(value) for value in mesh.extents],
        "bounds": [[float(value) for value in row] for row in mesh.bounds],
        "material_names": sorted(set(materials)),
        "watertight": bool(mesh.is_watertight),
    }


def _render_views(mesh, maximum_faces: int) -> list[Image.Image]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError("OBJ contains no vertices")
    if len(faces) > maximum_faces:
        faces = faces[evenly_spaced_indices(len(faces), maximum_faces)]
    centered = vertices - np.mean(mesh.bounds, axis=0)
    radius = max(float(np.max(mesh.extents)) / 2, np.finfo(float).eps)
    images = []
    for label, elevation, azimuth in VIEWS:
        figure = Figure(figsize=(5.12, 5.12), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        if len(faces):
            triangles = centered[faces]
            collection = Poly3DCollection(
                triangles,
                facecolor=(0.62, 0.68, 0.75, 1.0),
                edgecolor=(0.2, 0.2, 0.2, 0.16),
                linewidth=0.15,
            )
            axis.add_collection3d(collection)
        else:
            sample = evenly_spaced_indices(len(centered), min(len(centered), maximum_faces))
            points = centered[sample]
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)
        axis.set_zlim(-radius, radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(label)
        figure.tight_layout(pad=0.2)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        images.append(Image.fromarray(rgba[..., :3].copy(), mode="RGB"))
    return images


def process_obj(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    mesh, geometries = _load_obj(data)
    summary = _geometry_summary(mesh, geometries)
    views = _render_views(mesh, context.limits.max_obj_render_faces)
    prompt = (
        "Describe this 3D object exhaustively and objectively from the six labelled canonical views. "
        "Explain its overall shape, visible components, orientation, symmetries, openings, surfaces, and "
        "structural relationships. Do not infer the benchmark question and do not answer any task. "
        "Referenced textures or materials may be unavailable, so prioritize geometry.\nGeometry metadata:\n"
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    )
    return description_result(context.describe_images(views, prompt), "model/obj")
