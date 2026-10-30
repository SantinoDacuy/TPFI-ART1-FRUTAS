# TFI ART1 — Clasificación de frutas (Grupo 2)

Sistema de clasificación de frutas con la red de **Carpenter-Grossberg (ART1)**, en dos etapas:
**(1)** tipo de fruta (`citrico` / `tropical` / `carozo`) y **(2)** calidad de las manzanas identificadas
(`premium` / `comercial` / `descarte`). Materia Redes Neuronales, UADER FCyT.

> **Aclaraciones de alcance.** `carozo` es el alias histórico de **manzana** (datos y modelos lo conservan).
> `densidad_alta`, `es_pesada` y `textura_firme` son **aproximaciones visuales** (solidez del contorno, área × solidez,
> varianza del laplaciano); no miden densidad física, gramos ni firmeza mecánica. ART1 **no usa las etiquetas para
> aprender**: se asignan después para nombrar las categorías ([qué aprende ART1](docs/que_aprende_art1.md)).

El primer avance (red `CarGross.py`, datasets y modelos entregados) se conserva sin cambios en `datasets/`, `modelos/` y
`baseline_original/`. Qué se mejoró, con qué evidencia, qué límites siguen abiertos y qué necesita al equipo:
**[docs/informe_cambios.md](docs/informe_cambios.md)**. Dónde está cada variable: [diccionario](docs/diccionario_variables.md).

## Instalación

Requiere Python 3.10 o superior (verificado con 3.14.0 en Windows 11). Versiones **fijadas y verificadas** en un entorno
nuevo (ver `resultados/verificacion/`): OpenCV 5.0.0.93, NumPy 2.4.6, pytest 7.4.2.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt          # opencv-python-headless, numpy (versiones fijas)
pip install -r requirements-dev.txt      # además pytest, para correr las pruebas
python -m pytest tests -q                # 279 pruebas
python scripts/verificar_instalacion.py  # repite todo en un entorno virtual TEMPORAL y deja un registro
```

Las **fotos** (`datasets/processed_images_fruits/`, 339 imágenes) no están en el repositorio por tamaño. Sin ellas
funcionan la demo, `CarGross.py`, `probar_imagen.py` (con fotos propias o las de `ejemplos/`) y los experimentos de la
primera ronda; `scripts/extraer_mediciones.py`, `scripts/robustez.py` y `scripts/busqueda.py` las necesitan solo para
listar/medir (las mediciones ya están en `resultados/mediciones/`).

## Estructura

```
CarGross.py                  red ART1 (núcleo del primer avance), entrenamiento, etiquetado, modelos, CLI
validacion.py                esquemas de los CSV, validación de datasets, de modelos y del preprocesamiento
configuracion.py             parámetros por etapa (config/parametros_etapas.csv) y rutas
preprocesador.py             pipeline OpenCV: v1 del primer avance (intacta) + mejoras y descriptores de color
preprocesador_calidad.py     variables de calidad del primer avance (sin cambios)
extraccion.py                imagen -> vector binario, guiado por el preprocesamiento guardado (y escrito como CSV)
inferencia.py                inferencia de dos etapas (+ aprendizaje explícito) sobre una imagen
probar_imagen.py             clasificar una foto (CLI)        demo.py   demostración de punta a punta
scripts/                     curación segura, mediciones, búsqueda/comparación/final de la segunda ronda,
                             robustez, verificación de instalación, línea de base, ronda 1 (experimentos.py)
evaluacion/                  inventario y particiones por grupo (con bloque de informe), métricas, cadena de dos etapas
tests/                       pruebas (pytest)
datasets/  modelos/          ENTREGADOS en el primer avance (no se modifican)
baseline_original/           snapshot del código, datasets y modelos entregados (commit f01b4dd)
datasets_v2/  modelos_v2/    entregables nuevos por variante: principal/, alternativa_orden_canonico/, experimental_*/,
                             ronda1_medianas/ (respaldo de la primera ronda)
config/                      parámetros de entrenamiento por etapa, en CSV     resultados/   evidencia (ver su LEEME)
docs/                        informe de cambios, qué aprende ART1, diccionario de variables, figuras
ejemplos/                    5 fotos de ejemplo
```

## Uso

Todos los programas aceptan `--help`/`--ayuda`, informan los errores con un diagnóstico y terminan con **código de salida
distinto de cero** cuando algo falla: `0` ok · `1` corrida infructuosa · `2` argumentos · `3` entrada inválida o ausente ·
`4` error interno.

### 1. Demostración completa

```bash
python demo.py                  # corridas del primer avance + variante principal + fotos de ejemplo
python demo.py --sin-fotos
```

### 2. Una foto

```bash
python probar_imagen.py --imagen ejemplos/manzana_premium.jpg
python probar_imagen.py --imagen ejemplos/naranja.jpg
python probar_imagen.py --imagen ejemplos/manzana_comercial_defecto.jpg --legado    # modelos entregados (formato 1)
python probar_imagen.py --imagen ejemplos/auto.jpg                                  # objeto ajeno al dominio
```

El comportamiento normal es **inferencia: los pesos no cambian**. La salida distingue la *coincidencia ART1* (`|I∧T|/|I|`,
**no** es una probabilidad de acierto), la *pureza de entrenamiento* de la categoría (mezcla de etiquetas al entrenar,
**no** la confianza de esa foto) y la etiqueta, que puede faltar. `ejemplos/auto.jpg` muestra el límite: ART1 compara
bits, no reconoce frutas. Por defecto usa `modelos_v2/principal/`; un modelo guardado desde un CSV sin preprocesamiento
(tipo *vectores*) se rechaza con un mensaje claro.

**Qué esperar de las fotos de `ejemplos/`** (resultado real de la variante principal): las tres manzanas se clasifican
como manzana y reciben su calidad (`premium`, `comercial`, `descarte`), pero `naranja.jpg` también se clasifica como
manzana (etapa 1) y `auto.jpg` resuena con una categoría de manzana. Son los límites ya medidos: la separación
cítrico/manzana es débil y no hay rechazo de objetos ajenos al dominio (ver el informe).

**Aprendizaje explícito** (`--aprender`): plasticidad **en memoria** durante la ejecución (prototipo resonante: AND; sin
resonancia: categoría **sin etiqueta**). Solo se guarda con `--guardar-aprendizaje CARPETA`, en archivos nuevos:

```bash
python probar_imagen.py --imagen ejemplos/auto.jpg --aprender --guardar-aprendizaje salida_aprendizaje
```

### 3. Entrenar / agrupar sobre un CSV (`CarGross.py`)

```bash
python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.70 --epocas 1     # corrida del primer avance
python CarGross.py --csv datasets_v2/principal/calidad_entrenamiento.csv           # rho y demás: config/parametros_etapas.csv
python CarGross.py --csv datasets_v2/principal/morfologia_entrenamiento.csv --guardar-modelo modelos_v2/prueba.json
python CarGross.py --csv datasets_v2/principal/morfologia_informe.csv --inferir-con modelos_v2/principal/art1_morfologia.json
python CarGross.py --csv datasets/dataset_calidad_real.csv --migrar-modelo modelos/art1_calidad.json --guardar-modelo modelos_v2/calidad_migrado.json
```

* **Guardar desde un CSV**: el modelo lleva el **preprocesamiento exacto** que produjo el CSV. Se toma de
  `preprocesamiento_<etapa>.csv` junto al dataset (o de `--preprocesamiento RUTA`); se valida contra las variables del CSV
  (nombres y orden) **antes** de entrenar o escribir. `--preprocesamiento legado` solo se acepta si el CSV trae exactamente
  las variables del primer avance; sin sidecar ni declaración se guarda un modelo de **vectores** (no procesa fotos). Nunca
  se supone que un CSV fue generado con las constantes v1.
* Los datasets conocidos se validan con su esquema (encabezado, valores ausentes o no binarios, ids duplicados, etiquetas).
  Otros CSV: `--esquema generico --col-etiqueta COLUMNA`.
* `--orden aleatorio --semilla N` cambia el orden; `--epocas N` permite hasta N pasadas; un modelo existente no se pisa
  sin `--sobrescribir`; `--inferir-con` clasifica sin modificar pesos y valida las variables por nombre.

### 4. Curación segura de imágenes

```bash
python scripts/curar_dataset.py seleccionar          # escribe salida_curacion/ ; no toca imágenes ni datasets/
python scripts/curar_dataset.py eliminar --seleccion salida_curacion/seleccion_curacion.csv      # SIMULACIÓN
python scripts/curar_dataset.py eliminar --seleccion ... --ejecutar --destino salida_curacion/descartadas   # mueve
```

### 5. Reproducir los resultados

```bash
python scripts/baseline_original.py     # línea de base del primer avance
python scripts/extraer_mediciones.py    # (con las fotos) medidas continuas de las 339 fotos
python scripts/busqueda.py buscar       # segunda ronda: búsqueda con bloque de informe reservado
python scripts/comparar_ronda2.py       # comparación de variantes bajo las mismas particiones
python scripts/final_ronda2.py --sobrescribir   # datasets, preprocesamiento CSV, modelos y config de cada variante
python scripts/robustez.py --morfologia modelos_v2/principal/art1_morfologia.json --calidad modelos_v2/principal/art1_calidad.json --nombre principal
```

Todo es determinista (semillas fijas; GrabCut con semilla fija). La primera ronda (protocolo exploratorio previo) se
reproduce con `python scripts/experimentos.py todo`; **no la ejecute con `--sobrescribir`** salvo que quiera regenerar
`config/parametros_etapas.csv` y `modelos_v2/` de esa ronda (se respaldan en `ronda1_medianas/`).

### 6. Guion corto para mostrar cómo aprende la red

```bash
python CarGross.py --csv datasets_v2/principal/morfologia_entrenamiento.csv --rho 0.3     # pocos grupos grandes (4)
python CarGross.py --csv datasets_v2/principal/morfologia_entrenamiento.csv --rho 0.6     # valor de la variante principal (11)
python CarGross.py --csv datasets_v2/principal/morfologia_entrenamiento.csv --rho 0.8     # grupos más finos (25)
python CarGross.py --csv datasets_v2/principal/morfologia_informe.csv --inferir-con modelos_v2/principal/art1_morfologia.json
python probar_imagen.py --imagen ejemplos/manzana_descarte_severo.jpg
```

Subir ρ exige más parecido para entrar a un grupo y produce más grupos; con `--rho 0.95 --max-categorias 14` la corrida
termina como **infructuosa** (código 1) por proliferación. La penúltima orden evalúa 104 fotos que no se usaron para
entrenar (67,3 % de exactitud solo en la etapa 1; no es un conjunto externo intacto). Cada foto clasificada deja un
`diagnostico_clasificacion.png` (ignorado por git).

## Formatos

* **Entrada de la red**: CSV (ver `datasets_v2/LEEME.md`, `validacion.py`, [diccionario](docs/diccionario_variables.md)).
* **Preprocesamiento**: `preprocesamiento_<etapa>.csv` (metadato de entrada, en CSV): segmentador, normalización, opciones,
  y por variable la medida continua, el operador y el umbral exacto.
* **Modelo** (`modelos_v2/*/*.json`, formato 2): pesos, ρ, L, política de nulo, orden de variables, `preprocesamiento`,
  mapeo de categorías, procedencia, tipo (`fotografias` | `vectores`). Se **valida al cargar**. Los modelos entregados
  (formato 1) cargan en modo legado; `--migrar-modelo` crea un formato 2 con los mismos pesos.
* **Configuración**: `config/parametros_etapas.csv`.

## Resultados (resumen; léase con las limitaciones del informe)

Segunda ronda, 104 fotos de un **bloque de informe reservado** (no es un conjunto externo intacto: las 339 fotos ya se
habían usado). Detalle en [resultados/ronda2/comparacion.md](resultados/ronda2/comparacion.md):

| | receta original reentrenada | variante principal (compatible) | alternativa orden canónico | sin tope de categorías (experimental) |
|---|---|---|---|---|
| cadena total, 30 particiones internas (informe) | 34,2 % ± 5,2 | **63,0 % ± 4,3** | 62,5 % ± 6,0 | 68,4 % ± 4,0 |
| cadena total, ajuste final, una evaluación | 33,7 % | 52,9 % | 60,6 % | 72,1 % |
| categorías morfología / calidad | 16 / 17 | 11 / 11 | 11 / 10 | 45 / 16 |

La variante principal conserva las definiciones y los cortes morfológicos originales y el criterio de éxito del primer
avance (menos de 15 categorías). **Límites abiertos**: banco ya inspeccionado y filtrado, sin id de fruta física;
las etiquetas de calidad están confundidas con el escenario (mirando solo el borde de la foto se acierta tanto como con el
sistema); las predicciones cambian mucho al sustituir el fondo o variar la iluminación
([robustez](resultados/ronda2/robustez_principal.md)); la separación cítrico/manzana sigue débil.

## Referencias

[1] Sánchez-Sinencio, E. & Lau, C. (1992). *Artificial Neural Networks*. IEEE Press.
[2] Carpenter, G. A. & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern
recognition machine. *Computer Vision, Graphics, and Image Processing, 37*.
[3] Fausett, L. (1994). *Fundamentals of Neural Networks*. Prentice Hall.

El material de Lau exigido por la cátedra (`Lau.pp5.a.11`, `Lau.pp12.a.14`) no está en el repositorio: la equivalencia con
esa notación no se comprobó (ver `docs/que_aprende_art1.md`, sección 5).
