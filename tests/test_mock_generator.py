"""
Generador de 17 archivos .glb base sintéticos con mallas subdivididas y texturas PBR dummy embebidas
para probar el pipeline de principio a fin sin consumir créditos de Meshy AI.
"""

import io
import json
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.meshy_client import ALL_17_BASE_PARTS, SYMMETRIC_BASE_MAP, UNIQUE_PARTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pad4(data: bytes, pad_char: bytes = b"\x00") -> bytes:
    """Alinea un bloque de bytes a múltiplo de 4 bytes según la especificación glTF 2.0."""
    rem = len(data) % 4
    if rem == 0:
        return data
    return data + pad_char * (4 - rem)


def create_synthetic_pbr_textures(part_name: str, tex_size: int = 256) -> Tuple[bytes, bytes, bytes]:
    """
    Genera 3 imágenes PNG en memoria (BaseColor, MetallicRoughness ORM, NormalMap) para `part_name`.
    - En `body_front`, `body_rear` y `body_side_base`, pinta la franja superior (V > 0.58) con un color
      oscuro de baja luminancia (< 0.22) para que la heurística de cristales detecte `glass`.
    - En `body_rear`, pinta además una franja roja intensa (R > 0.5 y R > 1.8 * G) para que la heurística
      detecte automáticamente las calaveras `taillight`.
    """
    albedo = np.full((tex_size, tex_size, 4), (170, 175, 185, 255), dtype=np.uint8)
    orm = np.full((tex_size, tex_size, 4), (255, 110, 180, 255), dtype=np.uint8)
    normal = np.full((tex_size, tex_size, 4), (128, 128, 255, 255), dtype=np.uint8)

    # Crear patrón de relieve sintético en el mapa de normales
    y_idx, x_idx = np.meshgrid(np.arange(tex_size), np.arange(tex_size), indexing="ij")
    normal[:, :, 0] = (128 + (np.sin(x_idx * 0.25) * 18)).astype(np.uint8)
    normal[:, :, 1] = (128 + (np.cos(y_idx * 0.25) * 18)).astype(np.uint8)

    if part_name in ("body_front", "body_rear", "body_side_base"):
        # Color base de carrocería claro en la zona media
        albedo[:, :, :3] = (200, 205, 215)
        # En imágenes PIL/GLB, tanto si V se lee directo como invertido, pintamos una banda oscura
        # de cristal (RGB=(18, 24, 30), luminancia ~ 0.09 < 0.22) en V alto (filas superiores e inferiores
        # correspondientes al tercio superior de la malla) para garantizar correspondencia UV exacta.
        albedo[: int(tex_size * 0.42), :, :3] = (18, 24, 30)
        albedo[int(tex_size * 0.58) :, :, :3] = (18, 24, 30)

        if part_name == "body_rear":
            # Zona roja intensa para calaveras traseras (R=230 > 0.5*255, G=20 -> R > 1.8*G)
            r_start = int(tex_size * 0.42)
            r_end = int(tex_size * 0.58)
            albedo[r_start:r_end, : int(tex_size * 0.35), :3] = (230, 20, 20)

    def _to_png_bytes(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    return _to_png_bytes(albedo), _to_png_bytes(orm), _to_png_bytes(normal)


def build_subdivided_box_mesh(
    nx: int = 4,
    ny: int = 4,
    nz: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye una caja subdividida en 6 caras con suficientes polígonos en todas las alturas Z
    (más de 150 caras triangulares) para que la heurística de segmentación de `glass`, `trim` y `taillight`
    opere sobre geometría densa y realista.
    Mapea la coordenada UV.y directamente proporcional a la altura normalizada Z en [0, 1].
    """
    positions: List[Tuple[float, float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    indices: List[int] = []

    def _add_grid_face(
        origin: np.ndarray,
        dir_u: np.ndarray,
        dir_v: np.ndarray,
        normal_vec: Tuple[float, float, float],
        steps_u: int,
        steps_v: int,
        is_rear_taillight_zone: bool = False,
    ) -> None:
        base_idx = len(positions)
        for iv in range(steps_v + 1):
            fv = iv / float(steps_v)
            for iu in range(steps_u + 1):
                fu = iu / float(steps_u)
                pt = origin + fu * dir_u + fv * dir_v
                positions.append((float(pt[0]), float(pt[1]), float(pt[2])))
                normals.append(normal_vec)
                # z_norm en [0, 1] a partir de pt[2] que está en [-0.5, 0.5]
                z_norm = float(pt[2]) + 0.5
                if is_rear_taillight_zone and 0.35 <= z_norm <= 0.56:
                    # Apuntar UV a la región roja (u < 0.30, v=0.50) para las calaveras traseras
                    uvs.append((0.15, 0.50))
                elif z_norm > 0.58:
                    # Apuntar UV a la zona oscura de cristal (v=0.80)
                    uvs.append((fu, 0.80))
                else:
                    uvs.append((fu, 0.48))

        for iv in range(steps_v):
            for iu in range(steps_u):
                i0 = base_idx + iv * (steps_u + 1) + iu
                i1 = i0 + 1
                i2 = i0 + (steps_u + 1)
                i3 = i2 + 1
                indices.extend([i0, i1, i3, i0, i3, i2])

    # Cara frontal (-Y, inclinada ligeramente en Z para simular parabrisas/cofre)
    _add_grid_face(
        origin=np.array([-0.5, -0.5, -0.5]),
        dir_u=np.array([1.0, 0.0, 0.0]),
        dir_v=np.array([0.0, 0.0, 1.0]),
        normal_vec=(0.0, -0.85, 0.52),
        steps_u=nx,
        steps_v=nz,
    )
    # Cara trasera (+Y, con zona de calaveras y medallón)
    _add_grid_face(
        origin=np.array([0.5, 0.5, -0.5]),
        dir_u=np.array([-1.0, 0.0, 0.0]),
        dir_v=np.array([0.0, 0.0, 1.0]),
        normal_vec=(0.0, 0.85, 0.52),
        steps_u=nx,
        steps_v=nz,
        is_rear_taillight_zone=True,
    )
    # Cara izquierda (-X)
    _add_grid_face(
        origin=np.array([-0.5, 0.5, -0.5]),
        dir_u=np.array([0.0, -1.0, 0.0]),
        dir_v=np.array([0.0, 0.0, 1.0]),
        normal_vec=(-0.90, 0.0, 0.43),
        steps_u=ny,
        steps_v=nz,
    )
    # Cara derecha (+X)
    _add_grid_face(
        origin=np.array([0.5, -0.5, -0.5]),
        dir_u=np.array([0.0, 1.0, 0.0]),
        dir_v=np.array([0.0, 0.0, 1.0]),
        normal_vec=(0.90, 0.0, 0.43),
        steps_u=ny,
        steps_v=nz,
    )
    # Cara superior (+Z)
    _add_grid_face(
        origin=np.array([-0.5, -0.5, 0.5]),
        dir_u=np.array([1.0, 0.0, 0.0]),
        dir_v=np.array([0.0, 1.0, 0.0]),
        normal_vec=(0.0, 0.0, 1.0),
        steps_u=nx,
        steps_v=ny,
    )
    # Cara inferior (-Z)
    _add_grid_face(
        origin=np.array([-0.5, 0.5, -0.5]),
        dir_u=np.array([1.0, 0.0, 0.0]),
        dir_v=np.array([0.0, -1.0, 0.0]),
        normal_vec=(0.0, 0.0, -1.0),
        steps_u=nx,
        steps_v=ny,
    )

    pos_arr = np.array(positions, dtype=np.float32)
    norm_arr = np.array(normals, dtype=np.float32)
    uv_arr = np.array(uvs, dtype=np.float32)
    idx_arr = np.array(indices, dtype=np.uint16)
    return pos_arr, norm_arr, uv_arr, idx_arr


def write_synthetic_glb(filepath: Path, part_name: str) -> Path:
    """Escribe un archivo binario .glb glTF 2.0 válido con geometría y 3 texturas PBR embebidas."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pos_arr, norm_arr, uv_arr, idx_arr = build_subdivided_box_mesh(nx=4, ny=4, nz=6)
    albedo_png, orm_png, normal_png = create_synthetic_pbr_textures(part_name, tex_size=128)

    pos_bytes = _pad4(pos_arr.tobytes())
    norm_bytes = _pad4(norm_arr.tobytes())
    uv_bytes = _pad4(uv_arr.tobytes())
    idx_bytes = _pad4(idx_arr.tobytes())
    albedo_bytes = _pad4(albedo_png)
    orm_bytes = _pad4(orm_png)
    normal_bytes = _pad4(normal_png)

    blocks = [pos_bytes, norm_bytes, uv_bytes, idx_bytes, albedo_bytes, orm_bytes, normal_bytes]
    raw_lengths = [
        len(pos_arr.tobytes()),
        len(norm_arr.tobytes()),
        len(uv_arr.tobytes()),
        len(idx_arr.tobytes()),
        len(albedo_png),
        len(orm_png),
        len(normal_png),
    ]

    buffer_views = []
    offset = 0
    for blk, raw_len in zip(blocks, raw_lengths):
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": raw_len,
            }
        )
        offset += len(blk)

    bin_chunk = b"".join(blocks)
    pos_min = pos_arr.min(axis=0).tolist()
    pos_max = pos_arr.max(axis=0).tolist()

    gltf_json = {
        "asset": {"version": "2.0", "generator": "car-asset-pipeline-mock-generator"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": f"{part_name}_raw", "mesh": 0}],
        "meshes": [
            {
                "name": f"{part_name}_mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": f"{part_name}_mat",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
                "normalTexture": {"index": 2},
            }
        ],
        "textures": [
            {"source": 0},
            {"source": 1},
            {"source": 2},
        ],
        "images": [
            {"bufferView": 4, "mimeType": "image/png", "name": f"{part_name}_basecolor"},
            {"bufferView": 5, "mimeType": "image/png", "name": f"{part_name}_orm"},
            {"bufferView": 6, "mimeType": "image/png", "name": f"{part_name}_normal"},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": int(len(pos_arr)),
                "type": "VEC3",
                "min": [float(x) for x in pos_min],
                "max": [float(x) for x in pos_max],
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": int(len(norm_arr)),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,  # FLOAT
                "count": int(len(uv_arr)),
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": int(len(idx_arr)),
                "type": "SCALAR",
            },
        ],
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_chunk)}],
    }

    json_bytes = _pad4(json.dumps(gltf_json, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk), b"BIN\x00")

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(json_header)
        f.write(json_bytes)
        f.write(bin_header)
        f.write(bin_chunk)

    return filepath


def generate_all_17_mock_glbs_and_populate_zones(
    cache_dir: str | Path = PROJECT_ROOT / "output" / "_base_cache",
    zones_dir: str | Path = PROJECT_ROOT / "output" / "zones",
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Genera los 17 archivos .glb sintéticos en `cache_dir` (junto con `manifest.json`)
    y los clona en las 23 zonas requeridas en `zones_dir`.
    """
    c_dir = Path(cache_dir)
    z_dir = Path(zones_dir)
    c_dir.mkdir(parents=True, exist_ok=True)
    z_dir.mkdir(parents=True, exist_ok=True)

    base_paths: Dict[str, Path] = {}
    manifest: Dict[str, dict] = {}

    for part_name in ALL_17_BASE_PARTS:
        glb_file = c_dir / f"{part_name}.glb"
        write_synthetic_glb(glb_file, part_name)
        base_paths[part_name] = glb_file
        manifest[part_name] = {
            "preview_task_id": f"mock_preview_{part_name}",
            "preview_status": "SUCCEEDED",
            "refine_task_id": f"mock_refine_{part_name}",
            "refine_status": "SUCCEEDED",
            "local_path": str(glb_file),
            "status": "completed",
        }

    with open(c_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    zone_paths: Dict[str, Path] = {}
    for u_part in UNIQUE_PARTS:
        dst = z_dir / f"{u_part}.glb"
        shutil.copy2(base_paths[u_part], dst)
        zone_paths[u_part] = dst

    for sym_base, target_zones in SYMMETRIC_BASE_MAP.items():
        for z_name in target_zones:
            dst = z_dir / f"{z_name}.glb"
            shutil.copy2(base_paths[sym_base], dst)
            zone_paths[z_name] = dst

    return base_paths, zone_paths


def test_mock_generator_creates_17_base_and_23_zones(tmp_path: Path) -> None:
    """Prueba unitaria de pytest para verificar que se generan los 17 GLBs base y las 23 zonas."""
    cache_dir = tmp_path / "_base_cache"
    zones_dir = tmp_path / "zones"
    base_paths, zone_paths = generate_all_17_mock_glbs_and_populate_zones(cache_dir, zones_dir)

    assert len(base_paths) == 17
    assert len(zone_paths) == 23
    assert (cache_dir / "manifest.json").exists()

    for p in base_paths.values():
        assert p.exists() and p.stat().st_size > 100
    for p in zone_paths.values():
        assert p.exists() and p.stat().st_size > 100


if __name__ == "__main__":
    base_map, zone_map = generate_all_17_mock_glbs_and_populate_zones()
    print(f"Generados {len(base_map)} GLBs base en output/_base_cache y {len(zone_map)} zonas en output/zones.")
