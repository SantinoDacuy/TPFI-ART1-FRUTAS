# Comparacion de la version original y la mejorada

Generado por `python scripts/experimentos.py final` (ver docs/informe_cambios.md).

## Datos y condiciones

- Banco: 339 fotos (219 manzanas, 60 naranjas, 60 bananas). Son las que quedaron tras la curacion del primer avance: **ya estan filtradas**, el resultado es optimista respecto del uso real.
- Particion principal (semilla 0), por grupo de fotos tomadas con <= 3 s de diferencia (proxy de fruta/rafaga; NO hay id de fruta fisica ni de sesion): entrenamiento 190, validacion 71, prueba 78. No se demuestra independencia entre fotos de la misma sesion.
- Parametros de la version mejorada: rho morfologia 0.55, rho calidad 0.45 (elegidos en validacion), pasadas hasta 10, desempate 'mayoritaria_global', umbrales de morfologia 'mediana' y de calidad 'mediana' calculados con el entrenamiento, segmentador v1, tiene_tallo 'protuberancia', tono 'circular', extras morfologia ['tono_calido', 'saturacion_alta'], extras calidad ['sin_oscurecimiento'].
- Original entregado: modelos de `modelos/` (rho 0.85 / 0.70), constantes del primer avance; entrenados con fotos que hoy estan en validacion y prueba, por lo que **sobre esas particiones sus resultados son optimistas** (reinferencia).
- Original reentrenado: la receta del primer avance (segmentacion v1, constantes, asignacion online, etapa 1 solo con manzanas good, rho 0.85 / 0.70, una pasada) entrenada solo con la particion de entrenamiento.

## A. Diagnostico sobre las 339 fotos (reinferencia: incluye fotos de entrenamiento)

| sistema | fotos | tipo correcto | F1 macro tipo | sin identificar (etapa 1) | manzanas derivadas a calidad | naranjas/bananas derivadas | calidad aislada: exactitud / F1 | cadena manzanas (tipo+calidad) | cadena total | referencia mayoritaria (tipo) |
|---|---|---|---|---|---|---|---|---|---|---|
| original entregado | 339 | 167/339 (49.3%) | 0.479 | 0 | 96/219 | 44/120 | 73.5% / 0.745 | 65/219 (29.7%) | 136/339 (40.1%) | 64.6% / F1 0.262 |
| mejorado (entrenado con la particion principal) | 339 | 255/339 (75.2%) | 0.716 | 0 | 170/219 | 35/120 | 74.9% / 0.760 | 128/219 (58.4%) | 213/339 (62.8%) | 64.6% / F1 0.262 |

No es una validacion externa: en el mejorado, las fotos de entrenamiento (190) estan incluidas.

## B. Particion principal: tres evaluaciones por separado

### Reinferencia sobre ENTRENAMIENTO (no es evaluacion)

| sistema | fotos | tipo correcto | F1 macro tipo | sin identificar (etapa 1) | manzanas derivadas a calidad | naranjas/bananas derivadas | calidad aislada: exactitud / F1 | cadena manzanas (tipo+calidad) | cadena total | referencia mayoritaria (tipo) |
|---|---|---|---|---|---|---|---|---|---|---|
| original entregado | 190 | 70/190 (36.8%) | 0.357 | 0 | 42/122 | 35/68 | 70.5% / 0.706 | 28/122 (23.0%) | 56/190 (29.5%) | 64.2% / F1 0.261 |
| original reentrenado | 190 | 145/190 (76.3%) | 0.753 | 0 | 86/122 | 9/68 | 69.7% / 0.671 | 53/122 (43.4%) | 112/190 (58.9%) | 64.2% / F1 0.261 |
| mejorado | 190 | 157/190 (82.6%) | 0.802 | 0 | 101/122 | 12/68 | 68.0% / 0.689 | 71/122 (58.2%) | 127/190 (66.8%) | 64.2% / F1 0.261 |

### VALIDACION (se uso para elegir rho)

| sistema | fotos | tipo correcto | F1 macro tipo | sin identificar (etapa 1) | manzanas derivadas a calidad | naranjas/bananas derivadas | calidad aislada: exactitud / F1 | cadena manzanas (tipo+calidad) | cadena total | referencia mayoritaria (tipo) |
|---|---|---|---|---|---|---|---|---|---|---|
| original entregado | 71 | 47/71 (66.2%) | 0.659 | 0 | 25/46 | 3/25 | 65.2% / 0.660 | 14/46 (30.4%) | 36/71 (50.7%) | 64.8% / F1 0.262 |
| original reentrenado | 71 | 39/71 (54.9%) | 0.527 | 0 | 28/46 | 14/25 | 63.0% / 0.578 | 20/46 (43.5%) | 31/71 (43.7%) | 64.8% / F1 0.262 |
| mejorado | 71 | 51/71 (71.8%) | 0.687 | 0 | 35/46 | 9/25 | 76.1% / 0.772 | 27/46 (58.7%) | 43/71 (60.6%) IC95 [44.0%, 75.8%] | 64.8% / F1 0.262 |

### PRUEBA (no intervino en ninguna decision)

| sistema | fotos | tipo correcto | F1 macro tipo | sin identificar (etapa 1) | manzanas derivadas a calidad | naranjas/bananas derivadas | calidad aislada: exactitud / F1 | cadena manzanas (tipo+calidad) | cadena total | referencia mayoritaria (tipo) |
|---|---|---|---|---|---|---|---|---|---|---|
| original entregado | 78 | 50/78 (64.1%) | 0.618 | 0 | 29/51 | 6/27 | 88.2% / 0.894 | 23/51 (45.1%) | 44/78 (56.4%) | 65.4% / F1 0.264 |
| original reentrenado | 78 | 43/78 (55.1%) | 0.548 | 0 | 29/51 | 13/27 | 72.5% / 0.712 | 21/51 (41.2%) | 35/78 (44.9%) | 65.4% / F1 0.264 |
| mejorado | 78 | 47/78 (60.3%) | 0.546 | 0 | 34/51 | 14/27 | 90.2% / 0.909 | 30/51 (58.8%) | 43/78 (55.1%) IC95 [42.9%, 68.2%] | 65.4% / F1 0.264 |

IC95: remuestreo de grupos de fotos (bootstrap), 2000 repeticiones; con pocos grupos es ancho.

### Matrices de confusion en PRUEBA

**Mejorado - etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 5 | 0 | 8 | 0 |
| tropical | 0 | 8 | 6 | 0 |
| carozo | 14 | 3 | 34 | 0 |

**Mejorado - etapa 2 aislada (todas las manzanas reales de prueba)**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 17 | 0 | 4 | 0 |
| comercial | 1 | 12 | 0 | 0 |
| descarte | 0 | 0 | 17 | 0 |

**Original reentrenado - etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 6 | 0 | 7 | 0 |
| tropical | 0 | 8 | 6 | 0 |
| carozo | 22 | 0 | 29 | 0 |

**Original reentrenado - etapa 2 aislada**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 19 | 0 | 2 | 0 |
| comercial | 7 | 6 | 0 | 0 |
| descarte | 5 | 0 | 12 | 0 |

## C. Variabilidad entre 30 particiones aleatorias por grupo (media +- desvio)

| sistema | particion | cadena total | cadena de manzanas | tipo exactitud | tipo F1 macro | calidad aislada F1 macro | manzanas derivadas (n) | naranjas/bananas derivadas (n) | categorias (morf/cal) | rho (morf/cal) |
|---|---|---|---|---|---|---|---|---|---|---|
| original reentrenado | entrenamiento | 54.8% +- 6.3 | 45.7% +- 13.2 | 66.9% +- 9.9 | 0.617 +- 0.082 | 0.713 +- 0.048 | 78.83 +- 23.04 | 15.13 +- 7.97 | 17.0 / 15.6 | 0.85 / 0.70 |
| original reentrenado | validacion | 36.7% +- 8.5 | 38.3% +- 12.9 | 48.4% +- 10.4 | 0.411 +- 0.121 | 0.625 +- 0.117 | 27.87 +- 10.87 | 19.27 +- 12.04 | 17.0 / 15.6 | 0.85 / 0.70 |
| original reentrenado | prueba | 45.2% +- 12.3 | 40.9% +- 13.9 | 57.7% +- 14.6 | 0.514 +- 0.143 | 0.638 +- 0.097 | 28.77 +- 9.41 | 11.47 +- 7.45 | 17.0 / 15.6 | 0.85 / 0.70 |
| mejorado | entrenamiento | 72.9% +- 5.3 | 71.3% +- 7.1 | 84.4% +- 4.2 | 0.791 +- 0.061 | 0.798 +- 0.055 | 109.40 +- 7.94 | 14.50 +- 5.00 | 31.1 / 16.5 | 0.65 / 0.53 |
| mejorado | validacion | 65.8% +- 8.2 | 69.4% +- 13.2 | 76.1% +- 6.3 | 0.717 +- 0.077 | 0.767 +- 0.132 | 41.33 +- 5.25 | 11.83 +- 6.45 | 31.1 / 16.5 | 0.65 / 0.53 |
| mejorado | prueba | 60.2% +- 7.0 | 63.5% +- 10.1 | 72.3% +- 8.9 | 0.653 +- 0.110 | 0.748 +- 0.131 | 39.30 +- 4.69 | 11.83 +- 6.26 | 31.1 / 16.5 | 0.65 / 0.53 |

Referencia 'predecir siempre la mayoritaria' (prueba): cadena de manzanas 44.3% +- 2.8; tipo 63.8% +- 6.0. La desviacion refleja la variacion entre particiones; las particiones comparten fotos, por lo que NO es un intervalo de confianza independiente.

## D. Categorias de los modelos mejorados (etiquetado con prototipos finales)

**morfologia** (rho 0.55, 26 categorias, variables ['es_esferica', 'cascara_rugosa', 'tiene_tallo', 'densidad_alta', 'pigmentacion_oscura', 'color_uniforme', 'tono_calido', 'saturacion_alta'])

| categoria | prototipo | muestras (reinferencia) | distribucion | etiqueta | estado |
|---|---|---|---|---|---|
| 0 | 00001010 | 5 | {'carozo': 5} | carozo | etiquetada |
| 1 | 00010100 | 5 | {'carozo': 5} | carozo | etiquetada |
| 2 | 01000000 | 1 | {'carozo': 1} | carozo | etiquetada |
| 3 | 10000010 | 1 | {'carozo': 1} | carozo | etiquetada |
| 4 | 00010001 | 2 | {'carozo': 2} | carozo | etiquetada |
| 5 | 01100000 | 30 | {'carozo': 11, 'tropical': 19} | tropical | etiquetada |
| 6 | 00011000 | 1 | {'tropical': 1} | tropical | etiquetada |
| 7 | 10000001 | 5 | {'carozo': 4, 'citrico': 1} | carozo | etiquetada |
| 8 | 00110000 | 1 | {'tropical': 1} | tropical | etiquetada |
| 9 | 10010010 | 18 | {'carozo': 18} | carozo | etiquetada |
| 10 | 01101000 | 31 | {'carozo': 28, 'tropical': 3} | carozo | etiquetada |
| 11 | 00001001 | 2 | {'carozo': 2} | carozo | etiquetada |
| 12 | 00010101 | 7 | {'carozo': 4, 'citrico': 2, 'tropical': 1} | carozo | etiquetada |
| 13 | 01100100 | 8 | {'carozo': 2, 'tropical': 6} | tropical | etiquetada |
| 14 | 10100000 | 1 | {'carozo': 1} | carozo | etiquetada |
| 15 | 00100110 | 5 | {'carozo': 5} | carozo | etiquetada |
| 16 | 11000010 | 1 | {'carozo': 1} | carozo | etiquetada |
| 17 | 00011001 | 2 | {'carozo': 1, 'tropical': 1} | carozo | etiquetada_por_desempate |
| 18 | 00010110 | 4 | {'carozo': 3, 'tropical': 1} | carozo | etiquetada |
| 19 | 00000000 | 1 | {'carozo': 1} | carozo | etiquetada |
| 20 | 10001010 | 5 | {'carozo': 5} | carozo | etiquetada |
| 21 | 11100001 | 5 | {'carozo': 5} | carozo | etiquetada |
| 22 | 01001110 | 8 | {'carozo': 7, 'citrico': 1} | carozo | etiquetada |
| 23 | 10110100 | 4 | {'carozo': 2, 'citrico': 2} | carozo | etiquetada_por_desempate |
| 24 | 10010101 | 36 | {'carozo': 8, 'citrico': 28} | citrico | etiquetada |
| 25 | 10101111 | 1 | {'citrico': 1} | citrico | etiquetada |

Muestras de entrenamiento sin resonancia con los prototipos finales: 0.

**calidad** (rho 0.45, 11 categorias, variables ['es_grande', 'es_pesada', 'textura_firme', 'sin_defectos_superficie', 'color_homogeneo', 'sin_oscurecimiento'])

| categoria | prototipo | muestras (reinferencia) | distribucion | etiqueta | estado |
|---|---|---|---|---|---|
| 0 | 001000 | 5 | {'descarte': 2, 'premium': 3} | premium | etiquetada |
| 1 | 000100 | 3 | {'comercial': 1, 'descarte': 1, 'premium': 1} | premium | etiquetada_por_desempate |
| 2 | 100000 | 2 | {'comercial': 2} | comercial | etiquetada |
| 3 | 000000 | 6 | {'comercial': 6} | comercial | etiquetada |
| 4 | 001100 | 15 | {'descarte': 8, 'premium': 7} | descarte | etiquetada |
| 5 | 000001 | 2 | {'comercial': 2} | comercial | etiquetada |
| 6 | 110000 | 17 | {'comercial': 2, 'descarte': 4, 'premium': 11} | premium | etiquetada |
| 7 | 000101 | 13 | {'comercial': 12, 'premium': 1} | comercial | etiquetada |
| 8 | 001010 | 24 | {'descarte': 14, 'premium': 10} | descarte | etiquetada |
| 9 | 110001 | 29 | {'comercial': 9, 'descarte': 2, 'premium': 18} | premium | etiquetada |
| 10 | 111110 | 6 | {'premium': 6} | premium | etiquetada |

Muestras de entrenamiento sin resonancia con los prototipos finales: 0.

