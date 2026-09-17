# Documentación de sesión de trabajo — TFI ART1 Frutas
**Grupo 2 — Redes Neuronales — UADER FCyT**
Fecha: 08/09/2026

---

## 1. Estado inicial del proyecto

Al comenzar la sesión, el repositorio `tfi-art1-frutas/` contenía:

```
tfi-art1-frutas/
├── CarGross.py              ← Red ART1 implementada (COMPLETA)
├── preprocesador.py         ← Pipeline OpenCV (COMPLETA, pero con bugs)
├── requirements.txt         ← opencv-python-headless>=4.8, numpy>=1.24
├── README.md
├── comparacion_pipeline.png
├── Processed Images_Fruits/ ← Dataset de calidad en la RAÍZ (mal ubicado)
│   ├── Bad Quality_Fruits/
│   ├── Good Quality_Fruits/
│   └── Mixed Qualit_Fruits/
└── datasets/
    ├── dataset_morfologia.csv           ← 150 frutas, 6 vars binarias ✅
    ├── dataset_calidad_sintetico_pruebas.csv  ← 62 filas sintéticas
    └── dataset_morfologia_sintetico_pruebas.csv
```

### Lo que YA estaba hecho (Entregable 1):
- **PPT con sonido** explicando ART1 (ya entregado a los profesores)
- **CarGross.py**: implementación completa de la red Carpenter-Grossberg con:
  - Aprendizaje rápido (L=2)
  - Test de vigilancia (rho configurable)
  - Reset de nodos
  - Manejo de corridas infructuosas (`ProliferacionError`)
  - Entrada/salida por CSV
  - Reporte de pureza por cluster
- **preprocesador.py**: pipeline de preprocesamiento con:
  - `segmentar()` — separación fruta/fondo
  - `centrar_y_escalar()` — normalización posición/tamaño (200x200px canvas)
  - `extraer_features()` — 6 features: aspecto, textura laplaciana, defecto convexidad, solidez, gris promedio, desvío hue
  - `binarizar()` — convierte valores continuos a 0/1
  - `procesar_imagen()` — función unificada
- **dataset_morfologia.csv**: 150 frutas reales (cítrico/tropical/carozo), 6 variables binarias

---

## 2. Lo que se necesitaba hacer

Según la consigna del TFI, faltaba:
- **Dataset real de calidad** (el sintético no cuenta para la consigna: mínimo 50 datos, 5 variables)
- Un dataset generado desde **imágenes reales** procesadas con el pipeline OpenCV
- **demo.py**: un script que muestre todo funcionando de una sola vez

La profesora Daniela confirmó por mail que el enfoque era correcto y señaló dos detalles ya resueltos en el código: centrado/escalado ✅ y color como metadato (desvío hue) ✅.

---

## 3. Reorganización del dataset

### Problema detectado
La carpeta `Processed Images_Fruits/` estaba en la **raíz** del proyecto. Era el dataset de imágenes reales de calidad recién descargado/descomprimido.

### Acción tomada
Se movió a `datasets/processed_images_fruits/` para mantener coherencia estructural:

```powershell
New-Item -ItemType Directory -Path "datasets\processed_images_fruits"
Move-Item -Path "Processed Images_Fruits\*" -Destination "datasets\processed_images_fruits\"
Remove-Item -Path "Processed Images_Fruits" -Recurse
```

### Estructura resultante

```
datasets/processed_images_fruits/
├── Bad Quality_Fruits/
│   ├── Apple_Bad/      → 1.141 imágenes
│   ├── Banana_Bad/     → 1.087 imágenes
│   ├── Guava_Bad/      → 1.129 imágenes
│   ├── Lime_Bad/       → 1.085 imágenes
│   ├── Orange_Bad/     → 1.159 imágenes
│   └── Pomegranate_Bad/→ 1.187 imágenes
├── Good Quality_Fruits/
│   ├── Apple_Good/     → 1.149 imágenes
│   ├── Banana_Good/    → 1.113 imágenes
│   ├── Guava_Good/     → 1.152 imágenes
│   ├── Lime_Good/      → 1.094 imágenes
│   ├── Orange_Good/    → 1.216 imágenes
│   └── Pomegranate_Good/→ 5.940 imágenes ⚠️ desbalanceado
└── Mixed Qualit_Fruits/   ← NOTA: typo en el nombre original, NO renombrar
    ├── Apple/          → 113 imágenes
    ├── Banana/         → 285 imágenes
    ├── Guava/          → 148 imágenes
    ├── Lemon/          → 278 imágenes
    ├── Orange/         → 125 imágenes
    └── Pomegranate/    → 125 imágenes
```

**Total: 19.526 imágenes** | Formato: `.jpg` | Resolución: ~192×256 o 256×256 px | RGB 24bpp

**Nota sobre el desbalance**: `Pomegranate_Good` tiene 5.940 imágenes vs ~1.100 del resto. No afecta nuestro trabajo porque solo usamos manzanas y hacemos muestreo controlado.

**Mapeo a clases de calidad** (decisión de diseño):
- `Apple_Good` → **Premium**
- `Apple` (Mixed) → **Comercial**
- `Apple_Bad` → **Descarte**

---

## 4. Diagnóstico del segmentador

### Por qué era crítico
`preprocesador.py` ya tenía implementado el pipeline completo, pero la función `segmentar()` **asumía fondo blanco**:

```python
# CÓDIGO ORIGINAL — PROBLEMÁTICO
fondo = (s < 30) & (v > 200)   # solo detecta blanco puro en HSV
```

Esto funcionaba con Fruits-360 (fondo blanco uniforme), pero las imágenes del dataset de calidad tienen **fondo variable** (madera, tela, superficie oscura — fotos de campo).

### Diagnóstico cuantitativo (`scripts/diagnostico_segmentacion.py`)

Se probó sobre 9 muestras reales (3 por clase de Apple):

| Imagen | Clase | Fondo blanco | Foreground | Estado |
|---|---|:---:|:---:|:---:|
| 20190809_115439.jpg | Apple_Good | 2.5% | 97.9% | ❌ FALLO |
| 20190809_115448.jpg | Apple_Good | 7.0% | 93.0% | ✅ OK |
| 20190809_115451.jpg | Apple_Good | 4.8% | 95.5% | ❌ FALLO |
| IMG20200728175856.jpg | Apple_Bad | 17.0% | 81.9% | ✅ OK |
| IMG20200728175907.jpg | Apple_Bad | 9.6% | 89.8% | ✅ OK |
| IMG20200728175908.jpg | Apple_Bad | 10.4% | 89.0% | ✅ OK |
| IMG20200728125930.jpg | Apple_Mixed | 0.0% | 100.0% | ❌ FALLO |
| IMG20200728125932.jpg | Apple_Mixed | 0.1% | 100.0% | ❌ FALLO |
| IMG20200728125933.jpg | Apple_Mixed | 0.1% | 100.0% | ❌ FALLO |

**Resultado: 4/9 OK** — inaceptable para producir features confiables.

---

## 5. Fix del segmentador — Otsu

### Solución implementada

Se reemplazó el segmentador HSV por uno basado en **umbral de Otsu + morfología adaptativa**. Otsu no asume ningún color de fondo: opera sobre el contraste global de la imagen en escala de grises.

**Nuevo `segmentar()` en `preprocesador.py`:**

```python
def segmentar(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (7, 7), 0)
    _, bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(bin_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    contornos, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(cleaned)
    if contornos:
        mayor = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara, [mayor], -1, 255, thickness=cv2.FILLED)
    return mascara
```

**Pipeline de Otsu:**
1. Conversión a grises
2. Suavizado gaussiano (kernel 7×7) para reducir ruido de textura
3. Umbral Otsu invertido (pixeles oscuros = fruta, claros = fondo)
4. `MORPH_OPEN` × 2 — elimina ruido pequeño
5. `MORPH_CLOSE` × 3 — rellena huecos internos (manchas, brillo)
6. Quedarse solo con el contorno más grande (descarta artefactos)

### Comparación validada (`scripts/test_otsu_segmentador.py`)

**Resultado: 8/9 OK con Otsu vs 4/9 con el original.**

---

## 6. Bug en `extraer_features()` — IndexError

### El error
Al correr el extractor por primera vez, todas las imágenes fallaban con:
```
invalid index to scalar variable
```

### Causa raíz
En `extraer_features()`, el cálculo de defectos de convexidad:

```python
# CÓDIGO ORIGINAL — BUGGY
prof_max = max(d[0][3] for d in defectos) / 256.0
```

`cv2.convexityDefects()` puede devolver un array de forma variable. Con contornos reales (no de fondo blanco), la forma del array no siempre es `(N, 1, 4)` indexable con `d[0][3]`.

### Fix aplicado

```python
# CÓDIGO CORREGIDO
try:
    defectos = cv2.convexityDefects(c, hull_idx)
    if defectos is not None and defectos.ndim == 3:
        # Forma esperada: (N, 1, 4) — acceso seguro via numpy
        prof_max = float(np.max(defectos[:, 0, 3])) / 256.0
        prof_defecto = prof_max / h
except (cv2.error, IndexError, ValueError):
    prof_defecto = 0.0  # contorno degenerado
```

**Resultado tras el fix: 0 errores en 90 imágenes.**

---

## 7. Script extractor — `scripts/extractor_calidad.py`

### Qué hace
Genera `datasets/dataset_calidad_real.csv` desde las imágenes reales de Apple.

### Flujo interno
```
Para cada clase (premium/comercial/descarte):
  1. Lista todas las imágenes en la carpeta correspondiente
  2. Toma muestra aleatoria de N imágenes (reproducible con rng-seed)
  3. Por cada imagen:
     a. cv2.imread()
     b. preprocesador.procesar_imagen() → segmentación + features + binarización
     c. Agrega columna calidad_real (etiqueta para validación, no va a la red)
  4. Escribe fila al CSV
```

### Mapeo de carpetas a etiquetas
| Carpeta | Etiqueta CSV |
|---|---|
| `Good Quality_Fruits/Apple_Good/` | `premium` |
| `Mixed Qualit_Fruits/Apple/` | `comercial` |
| `Bad Quality_Fruits/Apple_Bad/` | `descarte` |

### Uso
```bash
python scripts/extractor_calidad.py              # 30 imgs/clase por defecto
python scripts/extractor_calidad.py --n 25       # 25 imgs/clase
python scripts/extractor_calidad.py --rng-seed 7 # otra semilla aleatoria
```

### Resultado
```
datasets/dataset_calidad_real.csv
  - 90 filas (30 premium + 30 comercial + 30 descarte)
  - 6 variables binarias
  - Cumple consigna: ≥50 datos, ≥5 variables ✅
```

**Cabecera del CSV:**
```
fruta_id,es_esferica,cascara_rugosa,tiene_tallo,densidad_alta,pigmentacion_oscura,color_uniforme,calidad_real
PRM001,0,1,0,0,1,0,premium
COM001,0,1,0,0,1,0,comercial
DES001,1,1,0,0,1,0,descarte
...
```

---

## 8. Corridas de CarGross.py — Resultados

### Dataset 1: Morfología (`dataset_morfologia.csv`)
150 frutas, tipos: cítrico / tropical / carozo

| rho | Clusters formados | Observación |
|:---:|:---:|---|
| 0.70 | 12 | Cluster 0: carozo 75%, Cluster 7: cítrico 82% |
| 0.85 | 17 | Cluster 10: cítrico 92%, mayor fragmentación tropical |

### Dataset 2: Calidad (`dataset_calidad_real.csv`)
90 manzanas reales, clases: premium / comercial / descarte

| rho | Clusters | Clusters de descarte | Pureza descarte |
|:---:|:---:|:---:|:---:|
| 0.70 | 7 | Clusters 4, 5, 6 | 100% / 100% / 100% |
| 0.85 | 9 | Clusters 6, 7, 8 | 93% / 100% / 100% |
| 0.95 | 9 | Igual que 0.85 | → red ya convergió |

**Observación clave**: Los clusters de `descarte` son los más puros (hasta 100%) porque las manzanas dañadas tienen features muy distintos (textura rugosa, baja solidez, color no uniforme). Los clusters de `premium` y `comercial` se mezclan más — comparten forma esférica y difieren solo en detalles de superficie.

---

## 9. Script demo — `demo.py`

### Para qué sirve
Un solo comando que ejecuta **todo el pipeline** de punta a punta, para demostrar el trabajo a los profesores o a cualquier persona.

### Qué hace en orden
1. Verifica que `CarGross.py` y los datasets existen
2. Si `dataset_calidad_real.csv` no existe, lo genera automáticamente
3. Corre Etapa 1: morfología con rho=0.70 y rho=0.85
4. Corre Etapa 2: calidad con rho=0.70, rho=0.85 y rho=0.95
5. Imprime "DEMO COMPLETADO" con instrucciones de uso

### Cómo correrlo
```bash
# Activar entorno
venv\Scripts\activate

# Corrida completa (genera dataset si no existe)
python demo.py

# Solo las corridas (usa dataset existente, más rápido)
python demo.py --solo-corridas

# Con más imágenes por clase
python demo.py --n-calidad 40
```

---

## 10. Estado final del repositorio

```
tfi-art1-frutas/
├── CarGross.py                          ← Red ART1 (sin cambios)
├── preprocesador.py                     ← MODIFICADO: Otsu + bug fix
├── demo.py                              ← NUEVO: demo completo
├── requirements.txt                     ← sin cambios
├── datasets/
│   ├── processed_images_fruits/         ← MOVIDO desde raíz
│   │   ├── Bad Quality_Fruits/
│   │   ├── Good Quality_Fruits/
│   │   └── Mixed Qualit_Fruits/
│   ├── dataset_morfologia.csv           ← sin cambios (150 frutas)
│   └── dataset_calidad_real.csv         ← NUEVO (90 manzanas reales)
└── scripts/
    ├── extractor_calidad.py             ← NUEVO: generador del CSV de calidad
    ├── diagnostico_segmentacion.py      ← NUEVO: diagnóstico (solo para ref.)
    └── test_otsu_segmentador.py         ← NUEVO: comparación Otsu vs original
```

---

## 11. Decisiones de diseño relevantes

| Decisión | Justificación |
|---|---|
| Otsu en lugar de HSV para segmentación | El dataset de calidad tiene fondo variable (no blanco). Otsu opera por contraste global sin asumir color. |
| Muestra de 30 imgs/clase (no usar todas) | El balance lo controla el muestreo, no el dataset crudo. Pomegranate_Good tiene 5x más imágenes que otras clases — usar todas sesgaría la red. |
| NO renombrar "Mixed Qualit_Fruits" (typo) | El script referencia la carpeta tal cual está escrita. Renombrar manualmente rompe el código sin aportar nada. |
| `calidad_real` como columna metadata | `CarGross.py` detecta automáticamente las columnas no-binarias y las excluye de la entrada a la red. Las usa solo para calcular pureza en el reporte. |
| rng-seed=42 fijo en extractor | Garantiza reproducibilidad: mismo CSV en cualquier máquina. |
| `try/except` en `convexityDefects` | La API de OpenCV devuelve arrays de forma variable según la geometría del contorno. El try/except es obligatorio para robustez con imágenes reales. |

---

## 12. Comandos de referencia rápida

```bash
# Instalar dependencias
pip install -r requirements.txt

# Generar dataset de calidad desde imágenes
python scripts/extractor_calidad.py --n 30

# Correr red ART1 sobre morfología
python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.7

# Correr red ART1 sobre calidad
python CarGross.py --csv datasets/dataset_calidad_real.csv --rho 0.85

# Demo completo de todo
python demo.py --solo-corridas
```
