"""
Prueba End-to-End (E2E) del pipeline car-asset-pipeline:
1. Genera los 17 GLBs sintéticos con texturas PBR y puebla las 23 zonas (test_mock_generator).
2. Ejecuta Blender en modo headless/background (src/blender_assembler.py).
3. Valida el archivo .glb exportado con src/glb_validator.py verificando el 100% de las reglas del contrato.
4. Verifica adicionalmente el cliente Meshy AI con mocks HTTP y las utilidades de Atlas UV.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.atlas_builder import (
    BODY_ATLAS_CELLS,
    MECHANICAL_ATLAS_CELLS,
    build_body_atlas,
    build_mechanical_atlas,
    remap_uv_coordinates,
)
from src.cli import run_assemble
from src.glb_validator import (
    ALL_9_CONTRACT_MATERIALS,
    REQUIRED_5_MATERIALS,
    REQUIRED_23_NODES,
    parse_glb_json,
    validate_glb,
)
from src.meshy_client import ALL_17_BASE_PARTS, MeshyClient
from tests.test_mock_generator import generate_all_17_mock_glbs_and_populate_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_uv_remapping_formula() -> None:
    """Verifica que la transformación UV a celdas 4x4 y 2x2 sea matemáticamente exacta."""
    # En cuadrícula 4x4, celda (2, 1) y UV (0.5, 0.5) -> ((0.5+2)/4, (0.5+1)/4) = (0.625, 0.375)
    u4, v4 = remap_uv_coordinates(0.5, 0.5, col=2, row=1, grid_size=4)
    assert pytest.approx(u4) == 0.625
    assert pytest.approx(v4) == 0.375

    # En cuadrícula 2x2, celda (1, 0) y UV (0.25, 0.75) -> ((0.25+1)/2, (0.75+0)/2) = (0.625, 0.375)
    u2, v2 = remap_uv_coordinates(0.25, 0.75, col=1, row=0, grid_size=2)
    assert pytest.approx(u2) == 0.625
    assert pytest.approx(v2) == 0.375


def test_meshy_client_caching_and_symmetry(tmp_path: Path) -> None:
    """Verifica que MeshyClient ejecute Preview -> Refine -> Descarga para 17 piezas y clone a 23 zonas."""
    cache_dir = tmp_path / "_base_cache"
    zones_dir = tmp_path / "zones"

    # Crear un GLB mínimo de prueba para simular descarga de Meshy
    dummy_source_dir = tmp_path / "dummy_src"
    base_paths, _ = generate_all_17_mock_glbs_and_populate_zones(
        cache_dir=dummy_source_dir / "_base_cache",
        zones_dir=dummy_source_dir / "zones",
    )
    sample_glb_bytes = next(iter(base_paths.values())).read_bytes()

    client = MeshyClient(
        api_key="msy_test_dummy_key",
        cache_dir=cache_dir,
        zones_dir=zones_dir,
        max_workers=4,
        poll_interval=0.01,
    )

    def fake_request(method: str, url: str, headers=None, json=None, timeout=60, stream=False):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if method == "POST" and json and json.get("mode") == "preview":
            resp.json.return_value = {"result": "task_preview_123"}
        elif method == "POST" and json and json.get("mode") == "refine":
            resp.json.return_value = {"result": "task_refine_456"}
        elif method == "GET" and url.startswith("https://api.meshy.ai"):
            resp.json.return_value = {
                "status": "SUCCEEDED",
                "progress": 100,
                "model_urls": {"glb": "https://assets.meshy.ai/mock_part.glb"},
            }
        elif method == "GET":
            resp.iter_content.return_value = [sample_glb_bytes]
        return resp

    with patch("src.meshy_client.requests.request", side_effect=fake_request) as mock_req:
        populated = client.generate_all(spec_path=PROJECT_ROOT / "config" / "car_spec.json")
        assert len(populated) == 23
        assert len(list(cache_dir.glob("*.glb"))) == 17
        assert (cache_dir / "manifest.json").exists()
        first_call_count = mock_req.call_count
        assert first_call_count > 0

        # Segunda llamada: debe usar 100% la caché en manifest.json sin realizar llamadas HTTP adicionales
        populated_cached = client.generate_all(spec_path=PROJECT_ROOT / "config" / "car_spec.json")
        assert len(populated_cached) == 23
        assert mock_req.call_count == first_call_count


def test_full_pipeline_e2e_blender_and_validator(tmp_path: Path) -> None:
    """
    Prueba End-to-End completa:
    1. Genera los 17 GLBs sintéticos en un directorio temporal y las 23 zonas.
    2. Verifica la generación de los Atlas PBR 4x4 (4096x4096) y 2x2 (2048x2048).
    3. Ejecuta Blender en modo headless para ensamblar y exportar car_final.glb.
    4. Valida car_final.glb con glb_validator y verifica que el 100% de los checks pasen.
    """
    cache_dir = tmp_path / "_base_cache"
    zones_dir = tmp_path / "zones"
    output_glb = tmp_path / "car_final.glb"
    spec_path = PROJECT_ROOT / "config" / "car_spec.json"

    # Paso 1: Generar 17 GLBs mock y 23 zonas
    base_paths, zone_paths = generate_all_17_mock_glbs_and_populate_zones(
        cache_dir=cache_dir,
        zones_dir=zones_dir,
    )
    assert len(base_paths) == len(ALL_17_BASE_PARTS) == 17
    assert len(zone_paths) == len(REQUIRED_23_NODES) == 23

    # Paso 2: Verificar generación de Atlas PBR
    atlas_dir = tmp_path / "_atlases"
    mech_atlases = build_mechanical_atlas(cache_dir, atlas_dir, atlas_size=4096)
    body_atlases = build_body_atlas(cache_dir, atlas_dir, atlas_size=2048)

    assert len(MECHANICAL_ATLAS_CELLS) == 11
    assert len(BODY_ATLAS_CELLS) == 3
    for p in mech_atlases.values():
        assert p.exists() and p.stat().st_size > 1000
    assert body_atlases["orm"].exists()
    assert body_atlases["normal"].exists()

    # Paso 3: Ensamblar en Blender (headless)
    assembled_path = run_assemble(
        input_dir=zones_dir,
        output_glb=output_glb,
        spec_path=spec_path,
    )
    assert assembled_path.exists()

    # Paso 4: Validar binariamente el archivo .glb exportado
    report = validate_glb(assembled_path)
    print("\n" + report.format_summary())
    assert report.passed, f"El validador GLB reportó fallos:\n{report.format_summary()}"

    # Verificaciones adicionales sobre el JSON interno del GLB
    _, _, _, gltf = parse_glb_json(assembled_path)
    exported_nodes = {n["name"] for n in gltf.get("nodes", [])}
    assert exported_nodes == set(REQUIRED_23_NODES)

    exported_mats = {m["name"] for m in gltf.get("materials", [])}
    for req_mat in REQUIRED_5_MATERIALS:
        assert req_mat in exported_mats
    for mat_name in exported_mats:
        assert "." not in mat_name, f"Material con sufijo detectado: {mat_name}"
        assert mat_name in ALL_9_CONTRACT_MATERIALS
