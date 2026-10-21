"""Cliente concurrente con caché y simetría para Meshy AI v2 Text-to-3D."""

import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MESHY_API_URL = "https://api.meshy.ai/openapi/v2/text-to-3d"

UNIQUE_PARTS: List[str] = [
    "engine",
    "transmission",
    "cooling",
    "fuel_system",
    "exhaust",
    "electrical",
    "battery",
    "ac_climate",
    "steering",
    "body_front",
    "body_rear",
    "lights",
    "interior",
]

SYMMETRIC_BASE_MAP: Dict[str, List[str]] = {
    "suspension_base": ["front_suspension", "rear_suspension"],
    "brakes_base": ["front_brakes", "rear_brakes"],
    "wheel_base": [
        "wheel_front_left",
        "wheel_front_right",
        "wheel_rear_left",
        "wheel_rear_right",
    ],
    "body_side_base": ["body_left", "body_right"],
}

ALL_17_BASE_PARTS: List[str] = UNIQUE_PARTS + list(SYMMETRIC_BASE_MAP.keys())


class MeshyClientError(Exception):
    """Excepción para errores de la API de Meshy AI."""


class MeshyClient:
    """Cliente concurrente para generar las 17 piezas base con Meshy AI v2 (Preview -> Refine -> Caché)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str | Path = "output/_base_cache",
        zones_dir: str | Path = "output/zones",
        max_workers: Optional[int] = None,
        poll_interval: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MESHY_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.zones_dir = Path(zones_dir)
        self.max_workers = int(os.getenv("MESHY_MAX_WORKERS", 2)) if max_workers is None else max_workers
        self.poll_interval = float(os.getenv("MESHY_POLL_INTERVAL", 10.0)) if poll_interval is None else poll_interval
        self.max_retries = int(os.getenv("MESHY_MAX_RETRIES", 25)) if max_retries is None else max_retries
        self.manifest_path = self.cache_dir / "manifest.json"
        self._lock = threading.Lock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("No se pudo leer manifest.json existente, iniciando nuevo: %s", exc)
        return {}

    def _save_manifest(self) -> None:
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _update_part_state(self, part_name: str, **kwargs: Any) -> None:
        with self._lock:
            entry = self.manifest.setdefault(part_name, {})
            entry.update(kwargs)
            entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._save_manifest()

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise MeshyClientError(
                "Falta MESHY_API_KEY. Configúrala en el archivo .env o pásala al inicializar MeshyClient."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_with_backoff(
        self,
        method: str,
        url: str,
        json_payload: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        backoff = 3.0
        for attempt in range(1, self.max_retries + 1):
            try:
                headers = self._headers() if url.startswith(MESHY_API_URL) else {}
                resp = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_payload,
                    timeout=60,
                    stream=stream,
                )
                is_rate_or_concurrency = False
                if resp.status_code == 429 or resp.status_code >= 500:
                    is_rate_or_concurrency = True
                elif resp.status_code in (400, 409):
                    resp_text = resp.text.lower()
                    if any(term in resp_text for term in ["concurrent", "queue", "rate limit", "too many"]):
                        is_rate_or_concurrency = True

                if is_rate_or_concurrency:
                    logger.warning(
                        "Límite de concurrencia o cola en Meshy AI (%d: %s). Reintentando en %.1fs (intento %d/%d)...",
                        resp.status_code,
                        resp.text[:100].strip(),
                        backoff,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 20.0)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise MeshyClientError(f"Error tras {self.max_retries} intentos en {url}: {exc}") from exc
                logger.warning(
                    "Excepción de red en %s (intento %d/%d): %s. Reintentando en %.1fs...",
                    url,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 20.0)
        raise MeshyClientError(f"Se agotaron los reintentos ({self.max_retries}) para {url}")

    def create_preview_task(self, prompt: str, target_polycount: int = 4000) -> str:
        """Inicia una tarea en modo preview en Meshy v2 y devuelve su task_id."""
        polycount = max(2000, min(8000, int(target_polycount)))
        payload = {
            "mode": "preview",
            "prompt": prompt,
            "ai_model": "latest",
            "topology": "triangle",
            "should_remesh": True,
            "target_polycount": polycount,
            "target_formats": ["glb"],
        }
        resp = self._request_with_backoff("POST", MESHY_API_URL, json_payload=payload)
        data = resp.json()
        task_id = data.get("result") or data.get("id")
        if not task_id:
            raise MeshyClientError(f"Respuesta de preview sin task_id válido: {data}")
        return str(task_id)

    def create_refine_task(self, preview_task_id: str) -> str:
        """Inicia una tarea en modo refine a partir de un preview_task_id y devuelve su task_id."""
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "enable_pbr": True,
            "texture_resolution": "2k",
            "target_formats": ["glb"],
        }
        resp = self._request_with_backoff("POST", MESHY_API_URL, json_payload=payload)
        data = resp.json()
        task_id = data.get("result") or data.get("id")
        if not task_id:
            raise MeshyClientError(f"Respuesta de refine sin task_id válido: {data}")
        return str(task_id)

    def poll_task(self, task_id: str, part_name: str, stage: str) -> Dict[str, Any]:
        """Realiza polling cada poll_interval segundos hasta que la tarea alcance SUCCEEDED."""
        task_url = f"{MESHY_API_URL}/{task_id}"
        while True:
            resp = self._request_with_backoff("GET", task_url)
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            progress = data.get("progress", 0)
            logger.info("[%s] (%s) task=%s status=%s (%s%%)", part_name, stage, task_id, status, progress)

            if status == "SUCCEEDED":
                return data
            if status in ("FAILED", "EXPIRED", "CANCELED"):
                error_msg = data.get("task_error", {}).get("message", "Sin detalle")
                raise MeshyClientError(
                    f"La tarea {stage} ({task_id}) para '{part_name}' falló con estado {status}: {error_msg}"
                )
            time.sleep(self.poll_interval)

    def download_glb(self, glb_url: str, dest_path: Path) -> Path:
        """Descarga el archivo .glb desde la URL entregada por Meshy."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = self._request_with_backoff("GET", glb_url, stream=True)
        temp_path = dest_path.with_suffix(".glb.part")
        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        temp_path.replace(dest_path)
        return dest_path

    def process_single_base_part(self, part_name: str, part_spec: Dict[str, Any]) -> Path:
        """Ejecuta el flujo Preview -> Refine -> Descarga con reanudación desde manifest.json."""
        cached_glb = self.cache_dir / f"{part_name}.glb"
        state = self.manifest.get(part_name, {})

        if state.get("status") == "completed" and cached_glb.exists() and cached_glb.stat().st_size > 0:
            logger.info("[%s] Ya existe en caché (%s). Saltando generación.", part_name, cached_glb)
            return cached_glb

        prompt = part_spec["prompt"]
        target_polycount = part_spec.get("target_polycount", 4000)

        # Etapa 1: Preview
        preview_task_id = state.get("preview_task_id")
        if not preview_task_id or state.get("preview_status") != "SUCCEEDED":
            if not preview_task_id:
                logger.info("[%s] Creando tarea preview...", part_name)
                preview_task_id = self.create_preview_task(prompt, target_polycount)
                self._update_part_state(
                    part_name,
                    preview_task_id=preview_task_id,
                    preview_status="IN_PROGRESS",
                    status="preview_polling",
                )
            self.poll_task(preview_task_id, part_name, "preview")
            self._update_part_state(part_name, preview_status="SUCCEEDED", status="preview_done")

        # Etapa 2: Refine
        refine_task_id = state.get("refine_task_id")
        if not refine_task_id:
            logger.info("[%s] Creando tarea refine desde preview=%s...", part_name, preview_task_id)
            refine_task_id = self.create_refine_task(preview_task_id)
            self._update_part_state(
                part_name,
                refine_task_id=refine_task_id,
                refine_status="IN_PROGRESS",
                status="refine_polling",
            )

        refine_data = self.poll_task(refine_task_id, part_name, "refine")
        glb_url = refine_data.get("model_urls", {}).get("glb")
        if not glb_url:
            raise MeshyClientError(f"[{part_name}] La tarea refine completó pero no contiene model_urls.glb")

        # Etapa 3: Descarga y caché
        logger.info("[%s] Descargando GLB refinado hacia %s...", part_name, cached_glb)
        self.download_glb(glb_url, cached_glb)
        self._update_part_state(
            part_name,
            refine_status="SUCCEEDED",
            glb_url=glb_url,
            local_path=str(cached_glb),
            status="completed",
        )
        return cached_glb

    def populate_zones_from_cache(self) -> Dict[str, Path]:
        """Distribuye las 17 piezas base de output/_base_cache/ hacia las 23 zonas en output/zones/."""
        self.zones_dir.mkdir(parents=True, exist_ok=True)
        populated: Dict[str, Path] = {}

        # Copiar las 13 piezas únicas
        for part_name in UNIQUE_PARTS:
            src = self.cache_dir / f"{part_name}.glb"
            if not src.exists():
                raise FileNotFoundError(f"Pieza base requerida no encontrada en caché: {src}")
            dst = self.zones_dir / f"{part_name}.glb"
            shutil.copy2(src, dst)
            populated[part_name] = dst

        # Clonar las 4 piezas simétricas en sus 10 destinos correspondientes
        for base_name, target_zones in SYMMETRIC_BASE_MAP.items():
            src = self.cache_dir / f"{base_name}.glb"
            if not src.exists():
                raise FileNotFoundError(f"Pieza base simétrica no encontrada en caché: {src}")
            for zone_name in target_zones:
                dst = self.zones_dir / f"{zone_name}.glb"
                shutil.copy2(src, dst)
                populated[zone_name] = dst

        return populated

    def generate_all(self, spec_path: str | Path = "config/car_spec.json") -> Dict[str, Path]:
        """Genera las 17 piezas base en paralelo (max_workers=4) y prepara las 23 zonas."""
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        base_parts: Dict[str, Dict[str, Any]] = spec.get("base_parts", {})
        missing_specs = [p for p in ALL_17_BASE_PARTS if p not in base_parts]
        if missing_specs:
            raise ValueError(f"Faltan definiciones en base_parts de car_spec.json: {missing_specs}")

        results: Dict[str, Path] = {}
        errors: Dict[str, Exception] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_part = {
                executor.submit(self.process_single_base_part, part_name, base_parts[part_name]): part_name
                for part_name in ALL_17_BASE_PARTS
            }
            for future in as_completed(future_to_part):
                part_name = future_to_part[future]
                try:
                    results[part_name] = future.result()
                except Exception as exc:
                    logger.error("Error procesando pieza base '%s': %s", part_name, exc)
                    errors[part_name] = exc

        if errors:
            failed = ", ".join(f"{k}: {v}" for k, v in errors.items())
            raise MeshyClientError(f"Falló la generación de {len(errors)} pieza(s) base: {failed}")

        return self.populate_zones_from_cache()
