# Diccionario de variables

Variables binarias que recibe la red. Las **comprometidas** del primer avance se conservan con su nombre, interpretación y
**corte original** en la variante principal; las marcadas **(adicional)** son descriptores cromáticos de la sugerencia
docente ("la información de color como metadato más para la red"). Valores de la columna *principal*: los guardados en
`datasets_v2/principal/preprocesamiento_<etapa>.csv` (la fuente de verdad es el modelo). `>=` / `<=` indican cuándo el bit
vale 1. Todo se mide sobre la fruta segmentada y centrada/escalada (canvas 200×200), salvo el área (foto original).

## Etapa 1 — morfología (`tipo_real`: `citrico`, `tropical`, `carozo` = manzana)

| variable | medida continua | qué mide realmente | bit = 1 si | primer avance | principal |
|---|---|---|---|---|---|
| `es_esferica` | `aspecto_esferico` | lado menor / lado mayor del rectángulo envolvente | `>=` | 0,80 | 0,80 |
| `cascara_rugosa` | `textura_laplaciano` | varianza del laplaciano de grises dentro de la máscara | `>=` | 150 | 150 |
| `tiene_tallo` | `defecto_convexidad` | **mismo criterio del primer avance**: mayor hendidura del contorno / alto (v1 la calculaba mal con OpenCV 5 y valía siempre 0). No es un detector de tallos | `>=` | 0,06 (siempre 0) | 0,06 |
| `densidad_alta` | `solidez` | **aproximación visual**: área / área del casco convexo. No es densidad física | `>=` | 0,90 | 0,90 |
| `pigmentacion_oscura` | `gris_promedio` | nivel de gris medio dentro de la máscara | `<=` | 130 | 130 |
| `color_uniforme` | `desvio_tono` | desvío **circular** del tono ponderado por saturación (v1: desvío lineal de H, que es cíclico: los rojos salían heterogéneos) | `<=` | 12 | 12 |
| `croma_alto` **(adicional)** | `croma_medio` | croma de Lab √(a²+b²) medio dentro de la máscara | `>=` | – | 45,9 (mediana de entrenamiento) |
| `naranja_dominante` **(adicional)** | `frac_naranja` | fracción de píxeles con color definido cuyo tono está en 15–45° | `>=` | – | 0,180 (mediana de entrenamiento) |

## Etapa 2 — calidad de manzanas (`calidad_real`: `premium`, `comercial`, `descarte`)

| variable | medida continua | qué mide realmente | bit = 1 si | primer avance | principal |
|---|---|---|---|---|---|
| `es_grande` | `area` | área de la máscara en **píxeles de la foto original** (no un tamaño físico calibrado) | `>=` | 10320,5 | 10198,5 |
| `es_pesada` | `volumen_proxy` | **aproximación visual**: área × solidez. No mide masa ni gramos | `>=` | 8567,8 | 9134,3 |
| `textura_firme` | `textura_laplaciano` | **aproximación visual**: baja varianza del laplaciano = superficie lisa. No mide firmeza mecánica | `<=` | 439,7 | 290,4 |
| `sin_defectos_superficie` | `defecto_score` | fracción de píxeles notablemente más oscuros que el promedio de la fruta | `<=` | 0,171 | 0,171 |
| `color_homogeneo` | `desvio_tono` | igual que `color_uniforme` | `<=` | 12,5 | 6,11 |
| `croma_alto` **(adicional)** | `croma_medio` | igual que en morfología | `>=` | – | 45,4 |

Los umbrales de calidad son **medianas del lote de entrenamiento** (la lógica de lote que el primer avance ya usaba, con el
valor exacto y calculado solo con entrenamiento); los de morfología son los cortes originales.

## Otras opciones disponibles (documentadas en `extraccion.py`; no están en la variante principal)

* Descriptores adicionales probados y **no adoptados** por empeorar la validación: `tono_calido`, `saturacion_alta`,
  `color_con_gradiente` / `color_sin_gradiente` (gradiente espacial del color), `tono_verdoso`, `tono_rojizo`,
  `tono_amarillento`, `rojo_dominante`, `amarillo_dominante`, `verde_dominante`, `sin_oscurecimiento`.
* Medición del color en el núcleo de la máscara o tras balance de blancos; normalización de iluminación previa
  (`estirar`, `gris_mundo`, `gamma_media`); `tiene_tallo` por protuberancia; umbrales por mediana: **experimentales**.

## Cómo leer los umbrales

* Un cambio de umbral cambia qué patrones son idénticos: es una decisión de preprocesamiento, no de la red
  (`que_aprende_art1.md`). Con umbrales por mediana (variante experimental) `es_esferica = 1` pasaría a significar "más
  redonda que la mediana del entrenamiento": por eso no se usa en la principal.
