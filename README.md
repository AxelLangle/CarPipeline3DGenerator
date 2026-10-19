# `car-asset-pipeline`

Herramienta CLI automatizada en Python y Blender (`bpy` en modo headless/background) para generar vehículos 3D modulares mediante la API de **Meshy AI Pro (v2)** y ensamblarlos, texturizarlos con Atlas PBR, normalizarlos y exportarlos en formato `.glb` cumpliendo estrictamente con el contrato técnico de **23 nodos** y **9 materiales**.

---

## Características Principales

1. **Optimización de Simetría y Créditos en Meshy AI (`src/meshy_client.py`):**
   - Genera únicamente **17 piezas base** en `output/_base_cache/` (13 piezas únicas + 4 piezas simétricas: `suspension_base`, `brakes_base`, `wheel_base`, `body_side_base`).
   - Flujo en dos etapas (`preview` -> `refine` con texturas PBR 2K), concurrencia con `ThreadPoolExecutor(max_workers=4)`, backoff exponencial ante HTTP 429/5xx y persistencia de estado en `manifest.json`.
   - Clona automáticamente las 17 piezas base hacia las **23 zonas anatómicas** en `output/zones/`.
2. **Empaquetado de Atlas PBR (`src/atlas_builder.py`):**
   - **Atlas `mechanical` 4x4 (`4096x4096` px):** Combina `Base Color`, `MetallicRoughness (ORM)` y `Normal` de las 11 piezas mecánicas base en celdas de `1024x1024` px y re-mapea los UVs de los 13 objetos mecánicos sin sufijos `.001`.
   - **Atlas `body` Híbrido 2x2 (`2048x2048` px):** Empaqueta `Normal` y `MetallicRoughness` de `body_front`, `body_rear` y `body_side_base`, manteniendo el `Base Color` en blanco puro `(1.0, 1.0, 1.0, 1.0)` (`Metallic = 0.5`, `Roughness = 0.25`) para permitir teñido dinámico en el motor destino sin perder relieve PBR.
3. **Segmentación Automática y Ensamblaje en Blender (`src/blender_assembler.py`):**
   - Segmentación heurística con `bmesh` sobre `body_*` para asignar automáticamente los materiales `glass`, `trim` y `taillight` según posición geométrica, normales y luminancia/color del Albedo original de Meshy AI.
   - Escala real en **metros** (`1.82m` ancho × `4.40m` largo × `1.45m` alto), orientación `-Y` forward en Blender (`+Z` forward al exportar con `+Y Up`), espejo en eje X con recálculo de normales hacia afuera para piezas derechas y exportación `.glb` **sin compresión Draco**.
4. **Validador Binario Puro (`src/glb_validator.py`):**
   - Inspecciona los chunks binarios (`glTF` v2 + `JSON`) con `struct` y `json` sin requerir interfaz gráfica.

---

## Estructura del Proyecto

```text
car-asset-pipeline/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── config/
│   └── car_spec.json             # Prompts, polycounts, bounding boxes en metros y coordenadas (-Y forward)
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Punto de entrada CLI con subcomandos: generate, assemble, validate, all
│   ├── meshy_client.py           # Cliente concurrente para Meshy AI v2 Text-to-3D (Preview -> Refine -> Caché)
│   ├── atlas_builder.py          # Empaquetador de texturas (Pillow/NumPy) para crear Atlas PBR (4x4 mechanical, 2x2 body)
│   ├── blender_assembler.py      # Script ejecutado por Blender (bpy) para importar, re-mapear UVs, segmentar cristales y exportar
│   └── glb_validator.py          # Validador binario de archivos .glb (inspecciona el chunk JSON sin requerir GUI)
└── tests/
    ├── test_mock_generator.py    # Genera 17 GLBs sintéticos con texturas PBR dummy para testear sin gastar créditos de Meshy
    └── test_pipeline_e2e.py      # Prueba End-to-End que corre mock -> assemble -> glb_validator
```

---

## Instalación y Uso

1. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

2. **Configurar variables de entorno:**
   Copia `.env.example` a `.env` y define tu `MESHY_API_KEY` (y opcionalmente `BLENDER_EXECUTABLE`).

3. **Comandos CLI disponibles:**
   - **Generar piezas con Meshy AI v2:**
     ```bash
     python -m src.cli generate
     ```
   - **Generar piezas sintéticas de prueba (sin gastar créditos):**
     ```bash
     python -m src.cli generate --mock
     ```
   - **Ensamblar en Blender (modo headless):**
     ```bash
     python -m src.cli assemble --input-dir output/zones --output-glb output/car_final.glb
     ```
   - **Validar el archivo `.glb` resultante:**
     ```bash
     python -m src.cli validate output/car_final.glb
     ```
   - **Flujo completo End-to-End:**
     ```bash
     python -m src.cli all --mock
     ```

4. **Ejecutar suite de pruebas automatizadas:**
   ```bash
   pytest tests/test_pipeline_e2e.py
   ```
