"""
Script de ensamblaje anatómico, re-mapeo UV a Atlas PBR, segmentación heurística de cristales/calaveras
y exportación a glTF Binary (.glb) ejecutado en Blender (modo headless/background).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Asegurar que la raíz del proyecto esté en sys.path cuando Blender ejecuta este script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bmesh  # type: ignore
import bpy  # type: ignore
from mathutils import Vector  # type: ignore

from src.atlas_builder import (
    BODY_ZONE_TO_CELL,
    MECHANICAL_ZONE_TO_CELL,
    build_body_atlas,
    build_mechanical_atlas,
    remap_uv_coordinates,
)


def reset_clean_scene() -> None:
    """Limpia por completo objetos, mallas, materiales e imágenes previos y configura unidades métricas."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def _purge_unused_materials(allowed_names: set) -> None:
    """Elimina cualquier material en bpy.data.materials cuyo nombre no esté en allowed_names."""
    for mat in list(bpy.data.materials):
        if mat.name not in allowed_names:
            bpy.data.materials.remove(mat, do_unlink=True)


def create_pbr_atlas_material(
    name: str,
    basecolor_path: Optional[Path],
    orm_path: Optional[Path],
    normal_path: Optional[Path],
    base_color_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    metallic_factor: float = 0.5,
    roughness_factor: float = 0.25,
) -> bpy.types.Material:
    """
    Crea un material Principled BSDF limpio con el nombre exacto `name` y conecta:
    - Base Color (opcional; para 'body' no se conecta textura y se fija blanco puro)
    - ORM / MetallicRoughness en Non-Color (Canal B -> Metallic, Canal G -> Roughness)
    - Normal Map en Non-Color -> nodo NormalMap -> Normal
    """
    existing = bpy.data.materials.get(name)
    if existing is not None:
        bpy.data.materials.remove(existing, do_unlink=True)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    bsdf.inputs["Base Color"].default_value = base_color_rgba
    bsdf.inputs["Metallic"].default_value = metallic_factor
    bsdf.inputs["Roughness"].default_value = roughness_factor
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    if basecolor_path is not None and basecolor_path.exists():
        tex_bc = nodes.new(type="ShaderNodeTexImage")
        tex_bc.location = (-350, 220)
        img_bc = bpy.data.images.load(str(basecolor_path), check_existing=True)
        img_bc.colorspace_settings.name = "sRGB"
        tex_bc.image = img_bc
        links.new(tex_bc.outputs["Color"], bsdf.inputs["Base Color"])

    if orm_path is not None and orm_path.exists():
        tex_orm = nodes.new(type="ShaderNodeTexImage")
        tex_orm.location = (-550, -40)
        img_orm = bpy.data.images.load(str(orm_path), check_existing=True)
        img_orm.colorspace_settings.name = "Non-Color"
        tex_orm.image = img_orm

        sep = nodes.new(type="ShaderNodeSeparateColor")
        sep.location = (-250, -40)
        links.new(tex_orm.outputs["Color"], sep.inputs["Color"])
        # En glTF MetallicRoughness: Green = Roughness, Blue = Metallic
        links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    if normal_path is not None and normal_path.exists():
        tex_norm = nodes.new(type="ShaderNodeTexImage")
        tex_norm.location = (-550, -320)
        img_norm = bpy.data.images.load(str(normal_path), check_existing=True)
        img_norm.colorspace_settings.name = "Non-Color"
        tex_norm.image = img_norm

        norm_map = nodes.new(type="ShaderNodeNormalMap")
        norm_map.location = (-200, -320)
        links.new(tex_norm.outputs["Color"], norm_map.inputs["Color"])
        links.new(norm_map.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def create_secondary_contract_materials() -> Dict[str, bpy.types.Material]:
    """
    Crea los materiales 'glass', 'trim', 'chrome' y 'taillight' con las propiedades exactas del contrato.
    """
    mats: Dict[str, bpy.types.Material] = {}

    # 1. glass: Base Color = (0.65, 0.85, 0.95, 0.30), Alpha = 0.30, Roughness = 0.05, blend_method = 'BLEND'
    if "glass" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["glass"], do_unlink=True)
    glass = bpy.data.materials.new(name="glass")
    glass.use_nodes = True
    if hasattr(glass, "blend_method"):
        try:
            glass.blend_method = "BLEND"
        except Exception:
            pass
    if hasattr(glass, "surface_render_method"):
        try:
            glass.surface_render_method = "BLENDED"
        except Exception:
            pass
    g_bsdf = glass.node_tree.nodes.get("Principled BSDF")
    if g_bsdf:
        g_bsdf.inputs["Base Color"].default_value = (0.65, 0.85, 0.95, 0.30)
        g_bsdf.inputs["Roughness"].default_value = 0.05
        if "Alpha" in g_bsdf.inputs:
            g_bsdf.inputs["Alpha"].default_value = 0.30
    mats["glass"] = glass

    # 2. trim: plásticos y molduras negras mate
    if "trim" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["trim"], do_unlink=True)
    trim = bpy.data.materials.new(name="trim")
    trim.use_nodes = True
    t_bsdf = trim.node_tree.nodes.get("Principled BSDF")
    if t_bsdf:
        t_bsdf.inputs["Base Color"].default_value = (0.06, 0.06, 0.07, 1.0)
        t_bsdf.inputs["Metallic"].default_value = 0.05
        t_bsdf.inputs["Roughness"].default_value = 0.75
    mats["trim"] = trim

    # 3. chrome: detalles cromados / rines
    if "chrome" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["chrome"], do_unlink=True)
    chrome = bpy.data.materials.new(name="chrome")
    chrome.use_nodes = True
    c_bsdf = chrome.node_tree.nodes.get("Principled BSDF")
    if c_bsdf:
        c_bsdf.inputs["Base Color"].default_value = (0.92, 0.94, 0.96, 1.0)
        c_bsdf.inputs["Metallic"].default_value = 1.0
        c_bsdf.inputs["Roughness"].default_value = 0.08
    mats["chrome"] = chrome

    # 4. taillight: calaveras traseras rojas
    if "taillight" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["taillight"], do_unlink=True)
    taillight = bpy.data.materials.new(name="taillight")
    taillight.use_nodes = True
    tl_bsdf = taillight.node_tree.nodes.get("Principled BSDF")
    if tl_bsdf:
        tl_bsdf.inputs["Base Color"].default_value = (0.85, 0.04, 0.04, 1.0)
        tl_bsdf.inputs["Roughness"].default_value = 0.15
        if "Emission Color" in tl_bsdf.inputs:
            tl_bsdf.inputs["Emission Color"].default_value = (0.45, 0.02, 0.02, 1.0)
        if "Emission Strength" in tl_bsdf.inputs:
            tl_bsdf.inputs["Emission Strength"].default_value = 0.4
    mats["taillight"] = taillight

    return mats


def import_and_normalize_single_zone(
    glb_path: Path,
    zone_name: str,
    target_size: List[float],
    target_location: List[float],
    mirror_x: bool,
) -> bpy.types.Object:
    """
    Importa un GLB de zona, une múltiples mallas si existen, elimina nodos Empty residuales,
    escala a target_size, aplica espejo X con recálculo de normales si mirror_x=True,
    posiciona en target_location y aplica rotación/escala para dejar scale=(1,1,1) y rotation=(0,0,0).
    """
    before_objs = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    imported_objs = [obj for obj in bpy.data.objects if obj not in before_objs]

    if not imported_objs:
        raise RuntimeError(f"El archivo GLB '{glb_path}' no importó ningún objeto para la zona '{zone_name}'.")

    # Limpiar jerarquías padre manteniendo la transformación mundial
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_objs:
        obj.select_set(True)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    mesh_objs = [obj for obj in imported_objs if obj.type == "MESH"]
    empty_objs = [obj for obj in imported_objs if obj.type != "MESH"]

    # Eliminar nodos Empty o cámaras/luces residuales
    for emp in empty_objs:
        bpy.data.objects.remove(emp, do_unlink=True)

    if not mesh_objs:
        raise RuntimeError(f"El archivo GLB '{glb_path}' no contiene ningún objeto MESH para '{zone_name}'.")

    # Unir si hay más de una malla
    bpy.ops.object.select_all(action="DESELECT")
    for m_obj in mesh_objs:
        m_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()

    main_obj = bpy.context.view_layer.objects.active
    main_obj.name = zone_name
    main_obj.data.name = zone_name

    # Aplicar rotación y escala previas a la normalización
    bpy.ops.object.select_all(action="DESELECT")
    main_obj.select_set(True)
    bpy.context.view_layer.objects.active = main_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Centrar la geometría en el origen local (0,0,0) para escalar y posicionar con precisión
    bm = bmesh.new()
    bm.from_mesh(main_obj.data)
    if bm.verts:
        xs = [v.co.x for v in bm.verts]
        ys = [v.co.y for v in bm.verts]
        zs = [v.co.z for v in bm.verts]
        min_v = Vector((min(xs), min(ys), min(zs)))
        max_v = Vector((max(xs), max(ys), max(zs)))
        center_v = 0.5 * (min_v + max_v)
        cur_dims = max_v - min_v

        for v in bm.verts:
            v.co -= center_v

        sx = target_size[0] / cur_dims.x if cur_dims.x > 1e-5 else 1.0
        sy = target_size[1] / cur_dims.y if cur_dims.y > 1e-5 else 1.0
        sz = target_size[2] / cur_dims.z if cur_dims.z > 1e-5 else 1.0

        for v in bm.verts:
            v.co.x *= sx
            v.co.y *= sy
            v.co.z *= sz

    bm.to_mesh(main_obj.data)
    bm.free()
    main_obj.data.update()

    # Espejeado en X para piezas derechas + recálculo de normales hacia afuera
    if mirror_x:
        main_obj.scale.x *= -1.0
        bpy.ops.object.select_all(action="DESELECT")
        main_obj.select_set(True)
        bpy.context.view_layer.objects.active = main_obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        bm_norm = bmesh.new()
        bm_norm.from_mesh(main_obj.data)
        bmesh.ops.recalc_face_normals(bm_norm, faces=bm_norm.faces)
        bm_norm.to_mesh(main_obj.data)
        bm_norm.free()
        main_obj.data.update()

    # Posicionar en target_location y asegurar scale=(1,1,1), rotation=(0,0,0)
    main_obj.location = Vector((target_location[0], target_location[1], target_location[2]))
    main_obj.rotation_mode = "XYZ"
    main_obj.rotation_euler = (0.0, 0.0, 0.0)
    main_obj.scale = (1.0, 1.0, 1.0)

    bpy.ops.object.select_all(action="DESELECT")
    main_obj.select_set(True)
    bpy.context.view_layer.objects.active = main_obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Para body_front aplicamos también location para que sus transformaciones sean 100% identidad
    # o conservamos sus vértices en espacio mundo con scale=(1,1,1) y rotation=(0,0,0)
    return main_obj


def remap_object_uvs_to_atlas_cell(obj: bpy.types.Object, col: int, row: int, grid_size: int) -> None:
    """
    Transforma todas las coordenadas UV del mapa activo del objeto a la celda (col, row)
    mediante uv.x = (uv.x + col) / grid_size y uv.y = (uv.y + row) / grid_size.
    """
    mesh = obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            uv = uv_layer[loop_idx].uv
            new_u, new_v = remap_uv_coordinates(float(uv[0]), float(uv[1]), col, row, grid_size)
            uv[0] = new_u
            uv[1] = new_v
    mesh.update()


def _load_albedo_pixels_for_body_part(albedo_path: Optional[Path]) -> Optional[Tuple[int, int, List[float]]]:
    """Carga los píxeles RGBA de una imagen de albedo usando bpy.data.images para muestreo rápido."""
    if albedo_path is None or not albedo_path.exists():
        return None
    img = bpy.data.images.load(str(albedo_path), check_existing=True)
    w, h = img.size[0], img.size[1]
    if w == 0 or h == 0:
        return None
    pixels = list(img.pixels)
    return w, h, pixels


def _sample_bpy_image_rgb(img_data: Optional[Tuple[int, int, List[float]]], u: float, v: float) -> Tuple[float, float, float]:
    """Muestrea (R, G, B) en [0..1] desde la tupla (width, height, flat_pixels) en espacio UV de Blender."""
    if img_data is None:
        return (0.5, 0.5, 0.5)
    w, h, pixels = img_data
    u_c = max(0.0, min(0.9999, u % 1.0 if (u < 0.0 or u > 1.0) else u))
    v_c = max(0.0, min(0.9999, v % 1.0 if (v < 0.0 or v > 1.0) else v))
    px = int(u_c * w)
    py = int(v_c * h)
    idx = (py * w + px) * 4
    if idx + 2 < len(pixels):
        return pixels[idx], pixels[idx + 1], pixels[idx + 2]
    return (0.5, 0.5, 0.5)


def segment_body_glass_trim_and_taillights(
    obj: bpy.types.Object,
    zone_name: str,
    body_mat: bpy.types.Material,
    glass_mat: bpy.types.Material,
    trim_mat: bpy.types.Material,
    taillight_mat: Optional[bpy.types.Material],
    original_albedo_path: Optional[Path],
) -> Dict[str, int]:
    """
    Aplica la heurística de segmentación en bmesh sobre `body_front`, `body_rear`, `body_left` y `body_right`
    ANTES de re-mapear los UVs al Atlas 2x2 (para muestrear con las UVs originales 0..1 del Albedo de Meshy):
    - Slot 0: body
    - Slot 1: glass
    - Slot 2: trim
    - Slot 3 (en body_rear): taillight
    """
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(body_mat)      # slot 0
    mesh.materials.append(glass_mat)     # slot 1
    mesh.materials.append(trim_mat)      # slot 2
    has_taillight = zone_name == "body_rear" and taillight_mat is not None
    if has_taillight and taillight_mat is not None:
        mesh.materials.append(taillight_mat)  # slot 3

    img_data = _load_albedo_pixels_for_body_part(original_albedo_path)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.verify()

    if not bm.verts:
        bm.free()
        return {"body": 0, "glass": 0, "trim": 0, "taillight": 0}

    zs = [v.co.z for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    z_min, z_max = min(zs), max(zs)
    y_min, y_max = min(ys), max(ys)
    height = max(1e-4, z_max - z_min)
    length_y = max(1e-4, y_max - y_min)

    glass_z_threshold = z_min + 0.58 * height
    trim_z_threshold = z_min + 0.14 * height

    glass_faces: List[Any] = []
    taillight_faces: List[Any] = []
    trim_faces: List[Any] = []

    for face in bm.faces:
        face.material_index = 0
        center = face.calc_center_median()
        normal = face.normal

        # Calcular UV promedio de la cara
        if face.loops:
            u_avg = sum(loop[uv_layer].uv.x for loop in face.loops) / len(face.loops)
            v_avg = sum(loop[uv_layer].uv.y for loop in face.loops) / len(face.loops)
        else:
            u_avg, v_avg = 0.5, 0.5

        r, g, b = _sample_bpy_image_rgb(img_data, u_avg, v_avg)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # 1. En body_rear, caras con predominancia roja (R > 0.5 y R > 1.8 * G) -> taillight (slot 3)
        if has_taillight and r > 0.5 and r > 1.8 * max(g, 0.01):
            face.material_index = 3
            taillight_faces.append(face)
            continue

        # 2. Heurística de cristales: tercio superior (z > z_min + 0.58 * height),
        # normal con componente vertical/lateral de parabrisas o ventanilla, y luminancia < 0.22
        has_window_normal = abs(normal.z) > 0.15 or abs(normal.x) > 0.25 or abs(normal.y) > 0.20
        if center.z > glass_z_threshold and has_window_normal and luminance < 0.22:
            face.material_index = 1
            glass_faces.append(face)
            continue

        # 3. Molduras bajas (trim) en el borde inferior
        if center.z < trim_z_threshold and luminance < 0.30:
            face.material_index = 2
            trim_faces.append(face)

    # Fallback geométrico obligatorio si la heurística encuentra menos de 10 caras para "glass"
    if len(glass_faces) < 10:
        candidates = sorted(
            [f for f in bm.faces if f.material_index == 0],
            key=lambda f: f.calc_center_median().z,
            reverse=True,
        )
        for f in candidates:
            c = f.calc_center_median()
            if c.z >= z_min + 0.52 * height:
                f.material_index = 1
                glass_faces.append(f)
                if len(glass_faces) >= max(12, len(bm.faces) // 8):
                    break
        # Si aún tiene menos de 10 caras, toma las 10 caras más altas de la pieza
        if len(glass_faces) < 10:
            for f in candidates[: min(10, len(candidates))]:
                if f.material_index != 1:
                    f.material_index = 1
                    glass_faces.append(f)

    # Garantizar al menos algunas caras de taillight en body_rear si no se detectaron por textura
    if has_taillight and len(taillight_faces) == 0:
        rear_candidates = sorted(
            [f for f in bm.faces if f.material_index == 0],
            key=lambda f: f.calc_center_median().y,
            reverse=True,
        )
        for f in rear_candidates[: min(8, len(rear_candidates))]:
            f.material_index = 3
            taillight_faces.append(f)

    # Garantizar presencia de trim en zonas bajas
    if len(trim_faces) == 0:
        bottom_candidates = sorted(
            [f for f in bm.faces if f.material_index == 0],
            key=lambda f: f.calc_center_median().z,
        )
        for f in bottom_candidates[: min(4, len(bottom_candidates))]:
            f.material_index = 2
            trim_faces.append(f)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return {
        "body": len(mesh.polygons) - len(glass_faces) - len(trim_faces) - len(taillight_faces),
        "glass": len(glass_faces),
        "trim": len(trim_faces),
        "taillight": len(taillight_faces),
    }


def preserve_and_rename_imported_material(obj: bpy.types.Object, exact_mat_name: str) -> bpy.types.Material:
    """
    Conserva el material importado de Meshy en `obj` con su árbol de nodos PBR completo y
    lo renombra exactamente a `exact_mat_name`, evitando colisiones `.001`.
    """
    existing = bpy.data.materials.get(exact_mat_name)
    mesh = obj.data

    imported_mat: Optional[bpy.types.Material] = None
    if mesh.materials and mesh.materials[0] is not None:
        imported_mat = mesh.materials[0]

    if existing is not None and existing != imported_mat:
        bpy.data.materials.remove(existing, do_unlink=True)

    if imported_mat is None:
        imported_mat = bpy.data.materials.new(name=exact_mat_name)
        imported_mat.use_nodes = True
    else:
        imported_mat.name = exact_mat_name

    mesh.materials.clear()
    mesh.materials.append(imported_mat)
    for poly in mesh.polygons:
        poly.material_index = 0
    mesh.update()
    return imported_mat


def assemble_vehicle_scene(
    input_dir: Path,
    output_glb: Path,
    spec_path: Path = PROJECT_ROOT / "config" / "car_spec.json",
) -> None:
    """Ejecuta el pipeline completo en Blender: Atlas PBR -> Importación de 23 nodos -> Materiales -> Exportación GLB."""
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    zones: List[Dict[str, Any]] = spec["zones"]
    atlas_dir = output_glb.parent / "_atlases"
    base_cache_dir = output_glb.parent / "_base_cache"
    texture_source_dir = base_cache_dir if base_cache_dir.exists() else input_dir

    # 1. Construir Atlas 4x4 para mechanical y Atlas 2x2 para body
    mech_atlas_paths = build_mechanical_atlas(texture_source_dir, atlas_dir, atlas_size=4096)
    body_atlas_paths = build_body_atlas(texture_source_dir, atlas_dir, atlas_size=2048)

    # 2. Limpiar escena de Blender y configurar unidades métricas
    reset_clean_scene()

    # 3. Crear los materiales de Atlas y materiales secundarios del contrato
    mech_mat = create_pbr_atlas_material(
        name="mechanical",
        basecolor_path=mech_atlas_paths["basecolor"],
        orm_path=mech_atlas_paths["orm"],
        normal_path=mech_atlas_paths["normal"],
        base_color_rgba=(1.0, 1.0, 1.0, 1.0),
        metallic_factor=0.8,
        roughness_factor=0.4,
    )

    # Regla 3: Material 'body' híbrido sin textura de Base Color (blanco puro 1,1,1,1, Metallic=0.5, Roughness=0.25)
    body_mat = create_pbr_atlas_material(
        name="body",
        basecolor_path=None,
        orm_path=body_atlas_paths["orm"],
        normal_path=body_atlas_paths["normal"],
        base_color_rgba=(1.0, 1.0, 1.0, 1.0),
        metallic_factor=0.5,
        roughness_factor=0.25,
    )

    secondary_mats = create_secondary_contract_materials()
    glass_mat = secondary_mats["glass"]
    trim_mat = secondary_mats["trim"]
    chrome_mat = secondary_mats["chrome"]
    taillight_mat = secondary_mats["taillight"]

    tire_mat: Optional[bpy.types.Material] = None

    # 4. Importar, normalizar, re-mapear UVs y asignar materiales a las 23 zonas
    for zone_cfg in zones:
        z_name = zone_cfg["name"]
        b_part = zone_cfg["base_part"]
        category = zone_cfg["category"]
        t_size = zone_cfg["target_size"]
        t_loc = zone_cfg["target_location"]
        mirror_x = bool(zone_cfg.get("mirror_x", False))

        zone_glb = input_dir / f"{z_name}.glb"
        if not zone_glb.exists():
            fallback_glb = texture_source_dir / f"{b_part}.glb"
            if fallback_glb.exists():
                zone_glb = fallback_glb
            else:
                raise FileNotFoundError(f"No se encontró el archivo GLB para la zona '{z_name}' en {zone_glb}")

        obj = import_and_normalize_single_zone(
            glb_path=zone_glb,
            zone_name=z_name,
            target_size=t_size,
            target_location=t_loc,
            mirror_x=mirror_x,
        )

        if category == "mechanical":
            col, row = MECHANICAL_ZONE_TO_CELL[z_name]
            remap_object_uvs_to_atlas_cell(obj, col, row, grid_size=4)
            obj.data.materials.clear()
            obj.data.materials.append(mech_mat)
            for poly in obj.data.polygons:
                poly.material_index = 0

        elif category == "body":
            albedo_key = f"albedo_{b_part}"
            albedo_path = body_atlas_paths.get(albedo_key)
            segment_body_glass_trim_and_taillights(
                obj=obj,
                zone_name=z_name,
                body_mat=body_mat,
                glass_mat=glass_mat,
                trim_mat=trim_mat,
                taillight_mat=taillight_mat if z_name == "body_rear" else None,
                original_albedo_path=albedo_path,
            )
            col, row = BODY_ZONE_TO_CELL[z_name]
            remap_object_uvs_to_atlas_cell(obj, col, row, grid_size=2)

        elif category == "wheel":
            if z_name == "wheel_front_left" or tire_mat is None:
                tire_mat = preserve_and_rename_imported_material(obj, "tire")
            else:
                obj.data.materials.clear()
                obj.data.materials.append(tire_mat)
                for poly in obj.data.polygons:
                    poly.material_index = 0
            # Añadir slot secundario 'chrome' en el centro del rin para soportar los 9 materiales del contrato
            obj.data.materials.append(chrome_mat)
            if len(obj.data.polygons) > 4:
                obj.data.polygons[0].material_index = 1

        elif category == "light":
            preserve_and_rename_imported_material(obj, "light")

        elif category == "interior":
            preserve_and_rename_imported_material(obj, "interior")

    # 5. Limpiar cualquier material huérfano o con sufijo `.001` que haya venido de los GLBs importados
    allowed_materials = {
        "body",
        "glass",
        "mechanical",
        "tire",
        "light",
        "interior",
        "trim",
        "chrome",
        "taillight",
    }
    _purge_unused_materials(allowed_materials)

    # 6. Exportar archivo .glb final sin compresión Draco y con +Y Up
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        export_yup=True,
        export_apply=True,
        export_draco_mesh_compression_enable=False,
    )
    print(f"[blender_assembler] Exportación exitosa: {output_glb}")


def parse_blender_cli_args() -> argparse.Namespace:
    """Extrae los argumentos pasados después de `--` al invocar `blender --background --python ... --`."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Ensamblador 3D en Blender para car-asset-pipeline")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directorio con los archivos .glb de entrada (ej. output/zones)",
    )
    parser.add_argument(
        "--output-glb",
        type=str,
        required=True,
        help="Ruta del archivo .glb final exportado (ej. output/car_final.glb)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=str(PROJECT_ROOT / "config" / "car_spec.json"),
        help="Ruta al archivo config/car_spec.json",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_blender_cli_args()
    assemble_vehicle_scene(
        input_dir=Path(args.input_dir).resolve(),
        output_glb=Path(args.output_glb).resolve(),
        spec_path=Path(args.spec).resolve(),
    )
