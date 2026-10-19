"""Empaquetador de texturas PBR y extractor de imágenes desde archivos GLB para Atlas 4x4 y 2x2."""

import io
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

MECHANICAL_ATLAS_CELLS: Dict[str, Tuple[int, int]] = {
    "engine": (0, 0),
    "transmission": (1, 0),
    "cooling": (2, 0),
    "fuel_system": (3, 0),
    "exhaust": (0, 1),
    "electrical": (1, 1),
    "battery": (2, 1),
    "ac_climate": (3, 1),
    "suspension_base": (0, 2),
    "brakes_base": (1, 2),
    "steering": (2, 2),
}

MECHANICAL_ZONE_TO_CELL: Dict[str, Tuple[int, int]] = {
    "engine": (0, 0),
    "transmission": (1, 0),
    "cooling": (2, 0),
    "fuel_system": (3, 0),
    "exhaust": (0, 1),
    "electrical": (1, 1),
    "battery": (2, 1),
    "ac_climate": (3, 1),
    "front_suspension": (0, 2),
    "rear_suspension": (0, 2),
    "front_brakes": (1, 2),
    "rear_brakes": (1, 2),
    "steering": (2, 2),
}

BODY_ATLAS_CELLS: Dict[str, Tuple[int, int]] = {
    "body_front": (0, 0),
    "body_rear": (1, 0),
    "body_side_base": (0, 1),
}

BODY_ZONE_TO_CELL: Dict[str, Tuple[int, int]] = {
    "body_front": (0, 0),
    "body_rear": (1, 0),
    "body_left": (0, 1),
    "body_right": (0, 1),
}


def remap_uv_coordinates(
    uv_x: float,
    uv_y: float,
    col: int,
    row: int,
    grid_size: int,
) -> Tuple[float, float]:
    """Transforma coordenadas UV al cuadrante (col, row) dentro de una cuadrícula grid_size x grid_size."""
    u_wrapped = uv_x % 1.0 if (uv_x < 0.0 or uv_x > 1.0) else uv_x
    v_wrapped = uv_y % 1.0 if (uv_y < 0.0 or uv_y > 1.0) else uv_y
    return (u_wrapped + float(col)) / float(grid_size), (v_wrapped + float(row)) / float(grid_size)


def _create_default_pbr_images(size: int = 512) -> Dict[str, Image.Image]:
    """Crea imágenes PBR por defecto en caso de que un GLB no tenga alguna textura."""
    basecolor = Image.new("RGBA", (size, size), (160, 165, 170, 255))
    # ORM glTF: R=Occlusion(255), G=Roughness(128), B=Metallic(180)
    orm = Image.new("RGBA", (size, size), (255, 128, 180, 255))
    # Normal map plano: (128, 128, 255)
    normal = Image.new("RGBA", (size, size), (128, 128, 255, 255))
    return {"basecolor": basecolor, "orm": orm, "normal": normal}


def extract_pbr_textures_from_glb(glb_path: str | Path) -> Dict[str, Image.Image]:
    """Extrae las texturas BaseColor, MetallicRoughness (ORM) y Normal directamente del binario GLB."""
    path = Path(glb_path)
    defaults = _create_default_pbr_images(256)
    if not path.exists():
        return defaults

    data = path.read_bytes()
    if len(data) < 20:
        return defaults

    magic, version, total_len = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        return defaults

    offset = 12
    json_dict: Optional[ dict ] = None
    bin_chunk: bytes = b""

    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_bytes = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            json_dict = json.loads(chunk_bytes.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk_bytes

    if not json_dict:
        return defaults

    textures = json_dict.get("textures", [])
    images = json_dict.get("images", [])
    buffer_views = json_dict.get("bufferViews", [])
    materials = json_dict.get("materials", [])

    def _load_image_by_tex_index(tex_idx: Optional[int]) -> Optional[Image.Image]:
        if tex_idx is None or tex_idx < 0 or tex_idx >= len(textures):
            return None
        img_idx = textures[tex_idx].get("source")
        if img_idx is None or img_idx < 0 or img_idx >= len(images):
            return None
        img_def = images[img_idx]
        bv_idx = img_def.get("bufferView")
        if bv_idx is not None and 0 <= bv_idx < len(buffer_views):
            bv = buffer_views[bv_idx]
            byte_offset = bv.get("byteOffset", 0)
            byte_length = bv.get("byteLength", 0)
            raw_bytes = bin_chunk[byte_offset : byte_offset + byte_length]
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                img.load()
                return img.convert("RGBA")
            except Exception:
                return None
        return None

    result = dict(defaults)
    if materials:
        mat = materials[0]
        pbr = mat.get("pbrMetallicRoughness", {})
        base_tex_info = pbr.get("baseColorTexture")
        mr_tex_info = pbr.get("metallicRoughnessTexture")
        norm_tex_info = mat.get("normalTexture")

        if base_tex_info is not None:
            loaded = _load_image_by_tex_index(base_tex_info.get("index"))
            if loaded is not None:
                result["basecolor"] = loaded

        if mr_tex_info is not None:
            loaded = _load_image_by_tex_index(mr_tex_info.get("index"))
            if loaded is not None:
                result["orm"] = loaded

        if norm_tex_info is not None:
            loaded = _load_image_by_tex_index(norm_tex_info.get("index"))
            if loaded is not None:
                result["normal"] = loaded

    return result


def _paste_cell_in_gltf_uv_space(
    atlas_img: Image.Image,
    cell_img: Image.Image,
    col: int,
    row: int,
    grid_size: int,
    cell_px: int,
) -> None:
    """
    Coloca una imagen en la celda (col, row) de un Atlas considerando que en Blender UV (0,0) es
    la esquina inferior izquierda, mientras que en imágenes PIL (0,0) es la esquina superior izquierda.
    """
    resized = cell_img.convert("RGBA").resize((cell_px, cell_px), Image.Resampling.BILINEAR)
    x_px = col * cell_px
    # Invertimos la fila en coordenadas de píxel para que row=0 corresponda a V in [0.0, 1/grid_size] en Blender
    y_px = (grid_size - 1 - row) * cell_px
    atlas_img.paste(resized, (x_px, y_px))


def build_mechanical_atlas(
    source_dir: str | Path,
    output_dir: str | Path,
    atlas_size: int = 4096,
) -> Dict[str, Path]:
    """
    Construye los 3 Atlas 4x4 (4096x4096, celdas de 1024x1024) para las 11 piezas mecánicas base:
    - mechanical_basecolor.png
    - mechanical_orm.png
    - mechanical_normal.png
    """
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid_size = 4
    cell_px = atlas_size // grid_size

    atlas_basecolor = Image.new("RGBA", (atlas_size, atlas_size), (140, 145, 150, 255))
    atlas_orm = Image.new("RGBA", (atlas_size, atlas_size), (255, 120, 190, 255))
    atlas_normal = Image.new("RGBA", (atlas_size, atlas_size), (128, 128, 255, 255))

    # Mapeo de respaldo por si source_dir es output/zones en lugar de output/_base_cache
    fallback_zone_names = {
        "suspension_base": "front_suspension",
        "brakes_base": "front_brakes",
    }

    for part_name, (col, row) in MECHANICAL_ATLAS_CELLS.items():
        glb_file = src_path / f"{part_name}.glb"
        if not glb_file.exists() and part_name in fallback_zone_names:
            glb_file = src_path / f"{fallback_zone_names[part_name]}.glb"

        pbr_imgs = extract_pbr_textures_from_glb(glb_file)
        _paste_cell_in_gltf_uv_space(atlas_basecolor, pbr_imgs["basecolor"], col, row, grid_size, cell_px)
        _paste_cell_in_gltf_uv_space(atlas_orm, pbr_imgs["orm"], col, row, grid_size, cell_px)
        _paste_cell_in_gltf_uv_space(atlas_normal, pbr_imgs["normal"], col, row, grid_size, cell_px)

    paths = {
        "basecolor": out_path / "mechanical_basecolor.png",
        "orm": out_path / "mechanical_orm.png",
        "normal": out_path / "mechanical_normal.png",
    }
    atlas_basecolor.save(paths["basecolor"], format="PNG", compress_level=1)
    atlas_orm.save(paths["orm"], format="PNG", compress_level=1)
    atlas_normal.save(paths["normal"], format="PNG", compress_level=1)
    return paths


def build_body_atlas(
    source_dir: str | Path,
    output_dir: str | Path,
    atlas_size: int = 2048,
) -> Dict[str, Path]:
    """
    Construye el Atlas 2x2 (2048x2048, celdas de 1024x1024) para body_front, body_rear y body_side_base:
    - body_orm.png
    - body_normal.png
    Además guarda los albedos originales por separado para que la heurística de cristales/calaveras
    pueda muestrear el color original de Meshy AI antes de aplicar el material 'body' blanco pintable.
    """
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    grid_size = 2
    cell_px = atlas_size // grid_size

    atlas_orm = Image.new("RGBA", (atlas_size, atlas_size), (255, 64, 128, 255))
    atlas_normal = Image.new("RGBA", (atlas_size, atlas_size), (128, 128, 255, 255))

    fallback_zone_names = {
        "body_side_base": "body_left",
    }

    paths: Dict[str, Path] = {}
    for part_name, (col, row) in BODY_ATLAS_CELLS.items():
        glb_file = src_path / f"{part_name}.glb"
        if not glb_file.exists() and part_name in fallback_zone_names:
            glb_file = src_path / f"{fallback_zone_names[part_name]}.glb"

        pbr_imgs = extract_pbr_textures_from_glb(glb_file)
        _paste_cell_in_gltf_uv_space(atlas_orm, pbr_imgs["orm"], col, row, grid_size, cell_px)
        _paste_cell_in_gltf_uv_space(atlas_normal, pbr_imgs["normal"], col, row, grid_size, cell_px)

        albedo_path = out_path / f"{part_name}_albedo.png"
        pbr_imgs["basecolor"].save(albedo_path, format="PNG", compress_level=1)
        paths[f"albedo_{part_name}"] = albedo_path

    paths["orm"] = out_path / "body_orm.png"
    paths["normal"] = out_path / "body_normal.png"
    atlas_orm.save(paths["orm"], format="PNG", compress_level=1)
    atlas_normal.save(paths["normal"], format="PNG", compress_level=1)
    return paths


def sample_albedo_rgb(albedo_array: np.ndarray, u: float, v: float) -> Tuple[float, float, float]:
    """
    Muestrea el color (R, G, B) normalizado en [0..1] desde un arreglo NumPy (H, W, C)
    usando coordenadas UV de Blender (donde v=0 es la parte inferior y v=1 la parte superior).
    """
    h, w = albedo_array.shape[:2]
    u_clamped = max(0.0, min(0.9999, u % 1.0 if (u < 0.0 or u > 1.0) else u))
    v_clamped = max(0.0, min(0.9999, v % 1.0 if (v < 0.0 or v > 1.0) else v))
    px_x = int(u_clamped * w)
    px_y = int((1.0 - v_clamped) * h)
    pixel = albedo_array[px_y, px_x]
    return float(pixel[0]) / 255.0, float(pixel[1]) / 255.0, float(pixel[2]) / 255.0
