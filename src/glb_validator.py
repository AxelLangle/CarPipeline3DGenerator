"""Validador binario automatizado de archivos .glb para el contrato de 23 nodos y 9 materiales."""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_23_NODES: List[str] = [
    "engine",
    "transmission",
    "cooling",
    "fuel_system",
    "exhaust",
    "electrical",
    "battery",
    "ac_climate",
    "front_suspension",
    "rear_suspension",
    "front_brakes",
    "rear_brakes",
    "steering",
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
    "body_front",
    "body_rear",
    "body_left",
    "body_right",
    "lights",
    "interior",
]

REQUIRED_5_MATERIALS: List[str] = [
    "body",
    "glass",
    "mechanical",
    "tire",
    "light",
]

SUPPORTED_OPTIONAL_MATERIALS: List[str] = [
    "interior",
    "trim",
    "chrome",
    "taillight",
]

ALL_9_CONTRACT_MATERIALS: List[str] = REQUIRED_5_MATERIALS + SUPPORTED_OPTIONAL_MATERIALS


@dataclass
class ValidationCheck:
    """Resultado individual de una verificación del contrato GLB."""

    name: str
    passed: bool
    details: str


@dataclass
class ValidationReport:
    """Reporte completo de validación binaria del archivo .glb."""

    glb_path: str
    checks: List[ValidationCheck] = field(default_factory=list)
    dimensions_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    node_names: List[str] = field(default_factory=list)
    material_names: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.checks) > 0 and all(c.passed for c in self.checks)

    def add_check(self, name: str, passed: bool, details: str) -> None:
        self.checks.append(ValidationCheck(name=name, passed=passed, details=details))

    def format_summary(self) -> str:
        lines = [
            f"=== Reporte de Validación GLB: {self.glb_path} ===",
        ]
        for idx, chk in enumerate(self.checks, 1):
            status_str = "PASS" if chk.passed else "FAIL"
            lines.append(f"  [{status_str}] {idx}. {chk.name}: {chk.details}")
        overall = "100% PASS" if self.passed else "FAIL"
        lines.append(f"Resultado Global: {overall} ({sum(c.passed for c in self.checks)}/{len(self.checks)} checks)")
        return "\n".join(lines)


def _quat_rotate_vec3(q: List[float], v: List[float]) -> List[float]:
    """Rota un vector 3D v por un cuaternión q = [x, y, z, w]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + qw * t + cross(q.xyz, t)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return [rx, ry, rz]


def _transform_point(
    pt: List[float],
    translation: List[float],
    rotation: List[float],
    scale: List[float],
    matrix: Optional[List[float]] = None,
) -> List[float]:
    """Aplica la transformación local de un nodo glTF a un punto 3D."""
    if matrix and len(matrix) == 16:
        # Matriz 4x4 column-major en glTF
        x, y, z = pt
        rx = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]
        ry = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]
        rz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
        return [rx, ry, rz]

    scaled = [pt[0] * scale[0], pt[1] * scale[1], pt[2] * scale[2]]
    rotated = _quat_rotate_vec3(rotation, scaled)
    return [
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    ]


def parse_glb_json(glb_path: str | Path) -> Tuple[bytes, int, int, Dict[str, Any]]:
    """
    Lee el encabezado binario del archivo .glb y extrae el chunk JSON.
    Devuelve (magic, version, total_length, json_dict).
    """
    path = Path(glb_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo GLB no existe: {path}")

    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"Archivo demasiado pequeño para ser un GLB válido ({len(raw)} bytes)")

    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    chunk_length, chunk_type = struct.unpack_from("<I4s", raw, 12)

    if chunk_type != b"JSON":
        raise ValueError(f"El primer chunk no es JSON, se encontró: {chunk_type!r}")

    json_bytes = raw[20 : 20 + chunk_length]
    gltf_dict = json.loads(json_bytes.decode("utf-8"))
    return magic, version, total_length, gltf_dict


def compute_node_world_bounds(
    node: Dict[str, Any],
    meshes: List[Dict[str, Any]],
    accessors: List[Dict[str, Any]],
) -> Optional[Tuple[List[float], List[float]]]:
    """Calcula el bounding box [min_x, min_y, min_z], [max_x, max_y, max_z] de un nodo con malla."""
    mesh_idx = node.get("mesh")
    if mesh_idx is None or mesh_idx < 0 or mesh_idx >= len(meshes):
        return None

    mesh = meshes[mesh_idx]
    primitives = mesh.get("primitives", [])
    if not primitives:
        return None

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    matrix = node.get("matrix")

    node_min = [float("inf"), float("inf"), float("inf")]
    node_max = [float("-inf"), float("-inf"), float("-inf")]
    found_valid = False

    for prim in primitives:
        pos_idx = prim.get("attributes", {}).get("POSITION")
        if pos_idx is None or pos_idx < 0 or pos_idx >= len(accessors):
            continue
        acc = accessors[pos_idx]
        acc_min = acc.get("min")
        acc_max = acc.get("max")
        if not acc_min or not acc_max or len(acc_min) < 3 or len(acc_max) < 3:
            continue

        # Transformar las 8 esquinas del AABB local para considerar rotación/escala/traslación
        corners = [
            [acc_min[0], acc_min[1], acc_min[2]],
            [acc_max[0], acc_min[1], acc_min[2]],
            [acc_min[0], acc_max[1], acc_min[2]],
            [acc_max[0], acc_max[1], acc_min[2]],
            [acc_min[0], acc_min[1], acc_max[2]],
            [acc_max[0], acc_min[1], acc_max[2]],
            [acc_min[0], acc_max[1], acc_max[2]],
            [acc_max[0], acc_max[1], acc_max[2]],
        ]
        for corner in corners:
            world_pt = _transform_point(corner, translation, rotation, scale, matrix)
            for axis in range(3):
                node_min[axis] = min(node_min[axis], world_pt[axis])
                node_max[axis] = max(node_max[axis], world_pt[axis])
        found_valid = True

    if not found_valid:
        return None
    return node_min, node_max


def validate_glb(glb_path: str | Path) -> ValidationReport:
    """Ejecuta las 7 validaciones técnicas del contrato sobre un archivo .glb."""
    report = ValidationReport(glb_path=str(glb_path))

    try:
        magic, version, total_length, gltf = parse_glb_json(glb_path)
    except Exception as exc:
        report.add_check("Magic Header & JSON Chunk", False, f"Error leyendo GLB: {exc}")
        return report

    # 1. Magic header igual a b'glTF' y versión 2
    header_ok = magic == b"glTF" and version == 2
    report.add_check(
        "Magic Header & Version 2",
        header_ok,
        f"magic={magic!r}, version={version}, bytes={total_length}",
    )

    # 2. Chunk JSON extraído válidamente
    json_ok = isinstance(gltf, dict) and "asset" in gltf
    report.add_check(
        "JSON Chunk Extraction",
        json_ok,
        f"glTF asset version={gltf.get('asset', {}).get('version', 'unknown')}",
    )

    # 3. Sin compresión Draco
    ext_used = gltf.get("extensionsUsed", []) or []
    ext_req = gltf.get("extensionsRequired", []) or []
    draco_name = "KHR_draco_mesh_compression"
    no_draco = (draco_name not in ext_used) and (draco_name not in ext_req)
    report.add_check(
        "Sin Compresión Draco",
        no_draco,
        f"extensionsUsed={ext_used}, extensionsRequired={ext_req}",
    )

    # 4. Exactamente los 23 nombres de nodos requeridos en minúsculas dentro de gltf["nodes"]
    nodes = gltf.get("nodes", [])
    node_names = [n.get("name", "") for n in nodes]
    report.node_names = node_names
    node_set = set(node_names)
    required_set = set(REQUIRED_23_NODES)
    missing_nodes = sorted(required_set - node_set)
    extra_nodes = sorted(node_set - required_set)
    nodes_ok = (len(missing_nodes) == 0) and (len(extra_nodes) == 0) and (len(nodes) == 23)
    if nodes_ok:
        nodes_detail = f"23/23 nodos obligatorios exactos presentes ({len(nodes)} nodos totales)"
    else:
        nodes_detail = f"Faltantes={missing_nodes}, Extras={extra_nodes}, Total={len(nodes)}"
    report.add_check("23 Nodos Obligatorios Exactos", nodes_ok, nodes_detail)

    # 5. Materiales obligatorios (body, glass, mechanical, tire, light) sin sufijos .001
    materials = gltf.get("materials", [])
    mat_names = [m.get("name", "") for m in materials]
    report.material_names = mat_names
    mat_set = set(mat_names)
    missing_req_mats = [m for m in REQUIRED_5_MATERIALS if m not in mat_set]
    suffixed_mats = [m for m in mat_names if "." in m or m not in ALL_9_CONTRACT_MATERIALS]
    mats_ok = (len(missing_req_mats) == 0) and (len(suffixed_mats) == 0)
    if mats_ok:
        mats_detail = f"Materiales válidos sin sufijos: {mat_names}"
    else:
        mats_detail = f"Faltantes obligatorios={missing_req_mats}, No permitidos/con sufijo={suffixed_mats}"
    report.add_check("Materiales Obligatorios y Sin Sufijos (.001)", mats_ok, mats_detail)

    # 6. Dimensiones totales del vehículo en metros (ancho [1.4, 2.5], alto [1.0, 2.5], largo [3.2, 5.8])
    meshes = gltf.get("meshes", [])
    accessors = gltf.get("accessors", [])
    global_min = [float("inf"), float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf"), float("-inf")]
    body_front_center_z: Optional[float] = None
    body_front_scale: Optional[List[float]] = None
    body_front_rotation: Optional[List[float]] = None

    for node in nodes:
        bounds = compute_node_world_bounds(node, meshes, accessors)
        if bounds is not None:
            n_min, n_max = bounds
            for axis in range(3):
                global_min[axis] = min(global_min[axis], n_min[axis])
                global_max[axis] = max(global_max[axis], n_max[axis])
            if node.get("name") == "body_front":
                body_front_center_z = 0.5 * (n_min[2] + n_max[2])
                body_front_scale = node.get("scale", [1.0, 1.0, 1.0])
                body_front_rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])

    if global_min[0] == float("inf"):
        report.add_check("Dimensiones en Metros", False, "No se encontraron accessors POSITION con min/max")
        report.add_check("Orientación Delantera (+Z) de body_front", False, "No se pudo calcular posición de body_front")
        return report

    # En glTF exportado con +Y Up: X = ancho, Y = alto, Z = largo (+Z forward)
    width_x = global_max[0] - global_min[0]
    height_y = global_max[1] - global_min[1]
    length_z = global_max[2] - global_min[2]
    report.dimensions_m = (width_x, height_y, length_z)

    width_ok = 1.4 <= width_x <= 2.5
    height_ok = 1.0 <= height_y <= 2.5
    length_ok = 3.2 <= length_z <= 5.8
    dims_ok = width_ok and height_ok and length_ok
    report.add_check(
        "Dimensiones Totales en Metros",
        dims_ok,
        f"Ancho(X)={width_x:.2f}m [1.4..2.5], Alto(Y)={height_y:.2f}m [1.0..2.5], Largo(Z)={length_z:.2f}m [3.2..5.8]",
    )

    # 7. body_front posicionado en el semieje delantero (+Z en glTF exportado) y con transformaciones aplicadas
    scale_applied = True
    if body_front_scale is not None:
        scale_applied = all(abs(s - 1.0) < 1e-3 for s in body_front_scale)
    rot_applied = True
    if body_front_rotation is not None:
        rot_applied = (
            abs(body_front_rotation[0]) < 1e-3
            and abs(body_front_rotation[1]) < 1e-3
            and abs(body_front_rotation[2]) < 1e-3
            and abs(abs(body_front_rotation[3]) - 1.0) < 1e-3
        )

    front_ok = (body_front_center_z is not None and body_front_center_z > 0.25) and scale_applied and rot_applied
    report.add_check(
        "Orientación y Transformación de body_front (+Z Forward)",
        front_ok,
        f"body_front center_z={body_front_center_z if body_front_center_z is not None else 'N/A':.3f}m (>0), "
        f"scale_applied={scale_applied}, rot_applied={rot_applied}",
    )

    return report
