"""Punto de entrada CLI para car-asset-pipeline con subcomandos: generate, assemble, validate, all."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.glb_validator import validate_glb
from src.meshy_client import MeshyClient

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_blender_executable(custom_path: Optional[str] = None) -> str:
    """Busca el ejecutable de Blender en custom_path, variable de entorno, PATH o rutas estándar del SO."""
    candidates: List[str] = []
    if custom_path:
        candidates.append(custom_path)
    env_blender = os.getenv("BLENDER_EXECUTABLE")
    if env_blender:
        candidates.append(env_blender)

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(which_blender)

    # Rutas estándar en Windows, macOS y Linux
    if sys.platform == "win32":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
        if pf.exists():
            for exe in sorted(pf.glob("Blender*/blender.exe"), reverse=True):
                candidates.append(str(exe))
    elif sys.platform == "darwin":
        candidates.append("/Applications/Blender.app/Contents/MacOS/Blender")
    else:
        candidates.extend(["/usr/bin/blender", "/snap/bin/blender"])

    for cand in candidates:
        if cand and Path(cand).is_file():
            return str(Path(cand).resolve())

    raise FileNotFoundError(
        "No se encontró el ejecutable de Blender. Agrégalo al PATH o define BLENDER_EXECUTABLE en .env."
    )


def run_generate(
    spec_path: Path,
    cache_dir: Path,
    zones_dir: Path,
    max_workers: int = 4,
    mock: bool = False,
) -> None:
    """Subcomando generate: genera las 17 piezas base (con Meshy AI o mock sintético) y puebla las 23 zonas."""
    if mock:
        from tests.test_mock_generator import generate_all_17_mock_glbs_and_populate_zones

        generate_all_17_mock_glbs_and_populate_zones(cache_dir=cache_dir, zones_dir=zones_dir)
        print(f"[generate] 17 piezas mock generadas en '{cache_dir}' y 23 zonas pobladas en '{zones_dir}'.")
        return

    client = MeshyClient(
        cache_dir=cache_dir,
        zones_dir=zones_dir,
        max_workers=max_workers,
    )
    populated = client.generate_all(spec_path=spec_path)
    print(f"[generate] Generación completada: {len(populated)} zonas listas en '{zones_dir}'.")


def run_assemble(
    input_dir: Path,
    output_glb: Path,
    spec_path: Path,
    blender_exe: Optional[str] = None,
) -> Path:
    """Subcomando assemble: ejecuta Blender en modo headless/background para ensamblar y exportar el .glb."""
    blender_bin = find_blender_executable(blender_exe)
    assembler_script = PROJECT_ROOT / "src" / "blender_assembler.py"

    cmd = [
        blender_bin,
        "--background",
        "--python",
        str(assembler_script),
        "--",
        "--input-dir",
        str(input_dir.resolve()),
        "--output-glb",
        str(output_glb.resolve()),
        "--spec",
        str(spec_path.resolve()),
    ]
    print(f"[assemble] Ejecutando Blender headless: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Blender falló con código de salida {proc.returncode}")

    if not output_glb.exists():
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise FileNotFoundError(f"Blender terminó pero no generó el archivo de salida: {output_glb}")

    print(f"[assemble] Archivo GLB ensamblado exitosamente en: {output_glb}")
    return output_glb


def run_validate(glb_path: Path) -> bool:
    """Subcomando validate: ejecuta el validador binario sobre el archivo .glb y muestra el reporte."""
    report = validate_glb(glb_path)
    print(report.format_summary())
    return report.passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-asset-pipeline",
        description="CLI automatizada para generar, ensamblar y validar vehículos 3D modulares (.glb).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: generate
    p_gen = subparsers.add_parser("generate", help="Genera las 17 piezas base con Meshy AI v2 y puebla las 23 zonas")
    p_gen.add_argument("--spec", type=Path, default=PROJECT_ROOT / "config" / "car_spec.json")
    p_gen.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "output" / "_base_cache")
    p_gen.add_argument("--zones-dir", type=Path, default=PROJECT_ROOT / "output" / "zones")
    p_gen.add_argument("--max-workers", type=int, default=4)
    p_gen.add_argument("--mock", action="store_true", help="Usa el generador sintético local en vez de Meshy AI")

    # Subcomando: assemble
    p_asm = subparsers.add_parser("assemble", help="Ensambla las 23 zonas y exporta el .glb en Blender headless")
    p_asm.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "output" / "zones")
    p_asm.add_argument("--output-glb", type=Path, default=PROJECT_ROOT / "output" / "car_final.glb")
    p_asm.add_argument("--spec", type=Path, default=PROJECT_ROOT / "config" / "car_spec.json")
    p_asm.add_argument("--blender-exe", type=str, default=None)

    # Subcomando: validate
    p_val = subparsers.add_parser("validate", help="Valida binariamente que un archivo .glb cumpla el contrato")
    p_val.add_argument("glb_path", type=Path, help="Ruta al archivo .glb a validar")

    # Subcomando: all
    p_all = subparsers.add_parser("all", help="Ejecuta generate -> assemble -> validate de principio a fin")
    p_all.add_argument("--spec", type=Path, default=PROJECT_ROOT / "config" / "car_spec.json")
    p_all.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "output" / "_base_cache")
    p_all.add_argument("--zones-dir", type=Path, default=PROJECT_ROOT / "output" / "zones")
    p_all.add_argument("--output-glb", type=Path, default=PROJECT_ROOT / "output" / "car_final.glb")
    p_all.add_argument("--max-workers", type=int, default=4)
    p_all.add_argument("--mock", action="store_true", help="Usa GLBs sintéticos locales para pruebas sin API Key")
    p_all.add_argument("--blender-exe", type=str, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        run_generate(
            spec_path=args.spec,
            cache_dir=args.cache_dir,
            zones_dir=args.zones_dir,
            max_workers=args.max_workers,
            mock=args.mock,
        )
        return 0

    if args.command == "assemble":
        run_assemble(
            input_dir=args.input_dir,
            output_glb=args.output_glb,
            spec_path=args.spec,
            blender_exe=args.blender_exe,
        )
        return 0

    if args.command == "validate":
        passed = run_validate(args.glb_path)
        return 0 if passed else 1

    if args.command == "all":
        run_generate(
            spec_path=args.spec,
            cache_dir=args.cache_dir,
            zones_dir=args.zones_dir,
            max_workers=args.max_workers,
            mock=args.mock,
        )
        run_assemble(
            input_dir=args.zones_dir,
            output_glb=args.output_glb,
            spec_path=args.spec,
            blender_exe=args.blender_exe,
        )
        passed = run_validate(args.output_glb)
        return 0 if passed else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
