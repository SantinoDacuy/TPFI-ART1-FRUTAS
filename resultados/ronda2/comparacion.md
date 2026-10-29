# Comparacion de variantes - segunda ronda

Generado por `python scripts/comparar_ronda2.py` (ver `docs/informe_cambios.md`, seccion de la segunda ronda).

## Condiciones

- Banco: 339 fotos, **ya inspeccionadas y reutilizadas** en la primera ronda; sin identificador de fruta fisica ni de sesion.
- Bloque de informe reservado (semilla 8, por grupos de fotos con <= 3 s de diferencia, estratificado por origen): 104 fotos (66 manzanas). Desarrollo: 235 fotos. **No es un conjunto externo intacto**: solo se evito usarlo para elegir en esta ronda.
- Todas las variantes se entrenan y se evaluan sobre las mismas particiones. Los umbrales dependientes de datos usan solo el entrenamiento; rho se elige en validacion interna; el informe no interviene en ninguna decision.
- Denominadores: ver el docstring de `scripts/comparar_ronda2.py` (N = fotos del informe; M = manzanas del informe).

## A. Ajuste final (todo 'desarrollo') evaluado UNA vez en el informe

| variante | rol | rho (morf/cal) | cadena total | cadena de manzanas | tipo exactitud (IC95) | tipo F1 macro | calidad aislada (exact. / F1) | calidad acoplada | manzanas derivadas | otras frutas derivadas | novedades / sin etiqueta (etapa 1) | categorias morf / cal (unitarias) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original_reentrenado | comparador: receta del primer avance | 0.85 / 0.7 | 35/104 (33.7%) IC [22.1%, 50.0%] | 18/66 (27.3%) | 47/104 (45.2%) [32.9%, 60.9%] | 0.413 | 66.7% / 0.639 | 18/30 (60.0%) | 30/66 (45.5%) | 11/38 (28.9%) | 0 / 0 | 16 (0) / 17 (1) |
| compatible_minima | solo correcciones y compromisos (sin descriptores adicionales) | 0.7 / 0.35 | 54/104 (51.9%) IC [35.8%, 66.4%] | 46/66 (69.7%) | 74/104 (71.2%) [51.9%, 87.4%] | 0.469 | 69.7% / 0.698 | 46/66 (69.7%) | 66/66 (100.0%) | 30/38 (78.9%) | 0 / 0 | 10 (1) / 8 (1) |
| principal | PRINCIPAL compatible con el primer avance | 0.6 / 0.45 | 55/104 (52.9%) IC [33.0%, 71.2%] | 43/66 (65.2%) | 70/104 (67.3%) [47.9%, 83.9%] | 0.537 | 74.2% / 0.740 | 43/58 (74.1%) | 58/66 (87.9%) | 21/38 (55.3%) | 4 / 0 | 11 (1) / 11 (2) |
| exp_sin_tope | experimental: sin el tope de 14 categorias del primer avance | 0.85 / 0.55 | 75/104 (72.1%) IC [55.2%, 86.1%] | 52/66 (78.8%) | 83/104 (79.8%) [62.4%, 91.8%] | 0.731 | 84.8% / 0.853 | 52/60 (86.7%) | 60/66 (90.9%) | 15/38 (39.5%) | 0 / 0 | 45 (12) / 16 (5) |
| exp_medianas | experimental: umbrales morfologicos por mediana | 0.3 / 0.45 | 59/104 (56.7%) IC [37.6%, 72.6%] | 49/66 (74.2%) | 76/104 (73.1%) [57.1%, 85.1%] | 0.505 | 74.2% / 0.740 | 49/66 (74.2%) | 66/66 (100.0%) | 28/38 (73.7%) | 0 / 0 | 9 (0) / 11 (2) |
| exp_protuberancia | experimental: tiene_tallo por protuberancia (sin anotaciones) | 0.6 / 0.45 | 60/104 (57.7%) IC [38.9%, 74.8%] | 45/66 (68.2%) | 75/104 (72.1%) [52.6%, 87.0%] | 0.587 | 74.2% / 0.740 | 45/60 (75.0%) | 60/66 (90.9%) | 18/38 (47.4%) | 0 / 0 | 13 (0) / 11 (2) |
| alt_orden_canonico | alternativa compatible: orden de presentacion canonico (determinista) | 0.55 / 0.35 | 63/104 (60.6%) IC [40.9%, 77.9%] | 52/66 (78.8%) | 75/104 (72.1%) [52.0%, 88.0%] | 0.555 | 80.3% / 0.815 | 52/64 (81.2%) | 64/66 (97.0%) | 26/38 (68.4%) | 0 / 0 | 11 (0) / 10 (1) |
| exp_comite | experimental: comite de 11 redes ART1 (cambia la arquitectura; costo 11x) | 0.55 / 0.45 | 63/104 (60.6%) IC [41.7%, 77.4%] | 49/66 (74.2%) | 77/104 (74.0%) [55.8%, 89.2%] | 0.610 | 75.8% / 0.781 | 49/63 (77.8%) | 63/66 (95.5%) | 23/38 (60.5%) | 0 / 0 | 14 (3) / 13 (2) |
| exp_segmentacion_v2 | experimental: segmentacion v2 | 0.45 / 0.35 | 40/104 (38.5%) IC [30.1%, 50.0%] | 26/66 (39.4%) | 74/104 (71.2%) [57.8%, 82.4%] | 0.589 | 42.4% / 0.452 | 26/60 (43.3%) | 60/66 (90.9%) | 24/38 (63.2%) | 0 / 0 | 12 (2) / 9 (1) |
| exp_normalizacion | experimental: estiramiento de contraste antes de medir (mas estable, menos exacto) | 0.55 / 0.35 | 39/104 (37.5%) IC [25.0%, 52.4%] | 36/66 (54.5%) | 52/104 (50.0%) [35.4%, 67.8%] | 0.271 | 74.2% / 0.745 | 36/49 (73.5%) | 49/66 (74.2%) | 34/38 (89.5%) | 0 / 0 | 13 (0) / 8 (1) |
| exp_medianas_ronda1 | experimental: receta de medianas de la ronda 1 (sin tope) | 0.75 / 0.55 | 57/104 (54.8%) IC [42.6%, 69.0%] | 39/66 (59.1%) | 69/104 (66.3%) [53.0%, 82.9%] | 0.596 | 80.3% / 0.804 | 39/51 (76.5%) | 51/66 (77.3%) | 13/38 (34.2%) | 7 / 0 | 42 (13) / 17 (2) |
| original_entregado | diagnostico historico (entrenado con fotos del informe: optimista) | (entregado: 0.85 / 0.70) | 44/104 (42.3%) IC [28.6%, 53.5%] | 21/66 (31.8%) | 55/104 (52.9%) [41.1%, 63.9%] | 0.509 | 68.2% / 0.669 | 21/32 (65.6%) | 32/66 (48.5%) | 10/38 (26.3%) | 0 / 0 | - |

Referencias mayoritarias en el informe (clase del entrenamiento): tipo 63.5% (F1 macro 0.259); cadena de manzanas 45.5%. IC95: remuestreo de grupos (2000 repeticiones); con pocos grupos es ancho y no implica significacion.

### Tamanos de las categorias del ajuste final (muestras de entrenamiento con prototipos finales)

| variante | etapa | categorias | unitarias | sin etiqueta | mediana | maximo | tamanos (descendente) |
|---|---|---|---|---|---|---|---|
| original_reentrenado | morfologia | 16 | 0 | 0 | 8.0 | 18 | [18, 17, 14, 14, 13, 12, 12, 8, 8, 6, 6, 6, 6, 5, 4, 2] |
| original_reentrenado | calidad | 17 | 1 | 0 | 9.0 | 21 | [21, 16, 15, 14, 12, 11, 11, 9, 9, 8, 6, 6, 5, 4, 3, 2, 1] |
| compatible_minima | morfologia | 10 | 1 | 0 | 11.0 | 73 | [73, 51, 47, 23, 14, 8, 7, 6, 5, 1] |
| compatible_minima | calidad | 8 | 1 | 0 | 20.5 | 33 | [33, 27, 23, 21, 20, 15, 13, 1] |
| principal | morfologia | 11 | 1 | 0 | 19.0 | 41 | [41, 40, 39, 31, 20, 19, 19, 12, 11, 2, 1] |
| principal | calidad | 11 | 2 | 0 | 15.0 | 29 | [29, 22, 22, 21, 20, 15, 14, 5, 3, 1, 1] |
| exp_sin_tope | morfologia | 45 | 12 | 1 | 3.0 | 25 | [25, 24, 19, 13, 11, 10, 9, 8, 8, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| exp_sin_tope | calidad | 16 | 5 | 2 | 3.0 | 29 | [29, 26, 23, 21, 16, 12, 11, 4, 2, 2, 2, 1, 1, 1, 1, 1] |
| exp_medianas | morfologia | 9 | 0 | 0 | 29.0 | 57 | [57, 46, 39, 29, 29, 19, 7, 5, 4] |
| exp_medianas | calidad | 11 | 2 | 0 | 15.0 | 29 | [29, 22, 22, 21, 20, 15, 14, 5, 3, 1, 1] |
| exp_protuberancia | morfologia | 13 | 0 | 0 | 13.0 | 48 | [48, 42, 39, 22, 19, 15, 13, 11, 9, 6, 4, 4, 3] |
| exp_protuberancia | calidad | 11 | 2 | 0 | 15.0 | 29 | [29, 22, 22, 21, 20, 15, 14, 5, 3, 1, 1] |
| alt_orden_canonico | morfologia | 11 | 0 | 0 | 14.0 | 51 | [51, 45, 38, 34, 19, 14, 13, 8, 6, 4, 3] |
| alt_orden_canonico | calidad | 10 | 1 | 1 | 11.0 | 35 | [35, 31, 27, 17, 11, 11, 9, 7, 4, 1] |
| exp_comite | morfologia | 14 | 3 | 0 | 19.5 | 50 | [50, 42, 30, 27, 27, 22, 17, 10, 7, 1, 1, 1] |
| exp_comite | calidad | 13 | 2 | 0 | 5.0 | 37 | [37, 29, 27, 16, 5, 3, 2, 1, 1] |
| exp_segmentacion_v2 | morfologia | 12 | 2 | 0 | 20.0 | 48 | [48, 46, 33, 27, 25, 25, 15, 7, 5, 2, 1, 1] |
| exp_segmentacion_v2 | calidad | 9 | 1 | 2 | 12.0 | 56 | [56, 38, 17, 16, 12, 8, 3, 2, 1] |
| exp_normalizacion | morfologia | 13 | 0 | 0 | 10.0 | 49 | [49, 45, 37, 33, 20, 17, 10, 8, 5, 3, 3, 3, 2] |
| exp_normalizacion | calidad | 8 | 1 | 0 | 12.5 | 54 | [54, 40, 24, 18, 7, 5, 4, 1] |
| exp_medianas_ronda1 | morfologia | 42 | 13 | 0 | 2.5 | 30 | [30, 21, 19, 18, 14, 12, 12, 11, 11, 10, 8, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| exp_medianas_ronda1 | calidad | 17 | 2 | 0 | 7.0 | 30 | [30, 19, 18, 16, 13, 10, 10, 9, 7, 5, 5, 3, 2, 2, 2, 1, 1] |

## B. Variabilidad entre 30 particiones internas (media +- desvio [minimo-maximo])

Entrenamiento/validacion salen de 'desarrollo'; el informe es el mismo bloque en todas. Cada particion entrena con ~70 % de desarrollo.

| variante | validacion: cadena total | validacion: cadena manzanas | informe: cadena total | informe: cadena manzanas | informe: tipo exactitud | informe: tipo F1 macro | informe: otras frutas derivadas (n) | categorias (morf/cal) | rho elegidos (morf/cal) |
|---|---|---|---|---|---|---|---|---|---|
| original_reentrenado | 41.2% +- 7.8 [27-59] | 35.0% +- 10.0 [18-49] | 34.2% +- 5.2 [21-44] | 33.8% +- 9.6 [17-48] | 48.1% +- 8.2 [35-62] | 0.410 +- 0.058 | 18.3 +- 5.9 | 15.2 / 14.1 | 0.85 / 0.70 |
| compatible_minima | 51.8% +- 8.6 [38-67] | 63.8% +- 8.7 [48-84] | 49.7% +- 3.7 [43-59] | 61.8% +- 9.6 [45-77] | 65.3% +- 4.4 [58-71] | 0.495 +- 0.054 | 23.1 +- 6.5 | 10.0 / 8.5 | 0.68 / 0.44 |
| principal | 65.2% +- 6.3 [56-79] | 68.7% +- 10.8 [45-90] | 63.0% +- 4.3 [55-73] | 72.5% +- 7.3 [56-88] | 76.0% +- 3.9 [67-83] | 0.654 +- 0.060 | 18.6 +- 4.0 | 10.0 / 9.8 | 0.58 / 0.44 |
| exp_sin_tope | 71.4% +- 7.1 [59-86] | 70.0% +- 8.5 [53-90] | 68.4% +- 4.0 [62-75] | 75.0% +- 5.8 [59-83] | 79.3% +- 2.9 [70-84] | 0.722 +- 0.047 | 15.1 +- 3.7 | 31.0 / 16.5 | 0.81 / 0.58 |
| exp_medianas | 56.6% +- 9.7 [43-77] | 69.5% +- 10.0 [42-92] | 59.5% +- 5.7 [50-72] | 73.5% +- 6.8 [59-83] | 72.5% +- 4.7 [62-85] | 0.552 +- 0.108 | 22.6 +- 6.5 | 9.6 / 9.8 | 0.33 / 0.44 |
| exp_protuberancia | 62.4% +- 9.3 [40-83] | 67.7% +- 10.1 [47-90] | 61.8% +- 5.4 [50-72] | 73.0% +- 6.4 [55-85] | 74.6% +- 4.4 [68-86] | 0.622 +- 0.100 | 18.6 +- 5.8 | 10.4 / 9.8 | 0.52 / 0.44 |
| alt_orden_canonico | 63.0% +- 7.8 [42-77] | 66.1% +- 9.2 [36-83] | 62.5% +- 6.0 [43-72] | 74.4% +- 10.3 [44-86] | 74.3% +- 6.2 [52-80] | 0.617 +- 0.082 | 20.2 +- 4.5 | 8.8 / 9.1 | 0.53 / 0.42 |
| exp_comite | 58.8% +- 10.9 [43-77] | 72.3% +- 8.0 [54-92] | 61.0% +- 4.6 [53-72] | 79.2% +- 4.6 [67-88] | 73.4% +- 3.6 [69-82] | 0.548 +- 0.104 | 25.0 +- 5.4 | 10.7 / 10.1 | 0.49 / 0.42 |
| exp_segmentacion_v2 | 55.2% +- 8.5 [41-78] | 51.5% +- 13.1 [27-84] | 48.1% +- 7.4 [35-63] | 51.2% +- 11.0 [35-70] | 67.6% +- 7.6 [51-80] | 0.566 +- 0.098 | 17.8 +- 6.8 | 10.1 / 9.5 | 0.47 / 0.41 |
| exp_normalizacion | 56.9% +- 9.3 [37-73] | 69.8% +- 9.0 [48-90] | 52.8% +- 5.3 [42-64] | 68.3% +- 5.6 [50-77] | 68.5% +- 6.1 [55-83] | 0.478 +- 0.118 | 25.5 +- 6.8 | 10.9 / 7.8 | 0.54 / 0.36 |
| exp_medianas_ronda1 | 65.9% +- 8.2 [47-80] | 68.7% +- 9.5 [50-86] | 57.1% +- 5.6 [45-66] | 63.0% +- 9.1 [39-74] | 68.5% +- 5.6 [56-79] | 0.605 +- 0.070 | 18.1 +- 5.0 | 34.3 / 14.1 | 0.72 / 0.52 |

Fallos (corridas infructuosas) por variante: original_reentrenado: 0, compatible_minima: 0, principal: 0, exp_sin_tope: 0, exp_medianas: 0, exp_protuberancia: 0, alt_orden_canonico: 0, exp_comite: 0, exp_segmentacion_v2: 0, exp_normalizacion: 0, exp_medianas_ronda1: 0.

## C. Sensibilidad al orden de presentacion (ajuste final, 30 ordenes aleatorios, informe)

| variante | cadena total: media +- desvio [min-max] | cadena de manzanas | tipo exactitud | categorias morf (min-max) | categorias cal (min-max) |
|---|---|---|---|---|---|
| original_reentrenado | 31.6% +- 3.3 [27-40] | 23.5% +- 6.9 [11-47] | 41.5% +- 4.7 [34-53] | 15-18 | 14-18 |
| compatible_minima | 47.0% +- 5.9 [36-56] | 60.7% +- 10.1 [42-74] | 63.2% +- 7.6 [47-72] | 9-13 | 7-9 |
| principal | 58.5% +- 8.5 [39-69] | 67.5% +- 9.0 [48-80] | 71.9% +- 8.2 [53-81] | 11-12 | 10-12 |
| exp_sin_tope | 67.7% +- 2.6 [63-74] | 77.5% +- 3.0 [71-80] | 77.3% +- 1.7 [75-82] | 45-45 | 16-17 |
| exp_medianas | 58.5% +- 5.0 [47-69] | 72.6% +- 6.1 [61-85] | 72.4% +- 4.5 [62-83] | 8-10 | 10-12 |
| exp_protuberancia | 60.3% +- 5.7 [40-72] | 72.6% +- 4.9 [64-82] | 75.4% +- 4.2 [62-84] | 13-14 | 10-12 |
| alt_orden_canonico | 60.6% +- 0.0 [61-61] | 78.8% +- 0.0 [79-79] | 72.1% +- 0.0 [72-72] | 11-11 | 10-10 |
| exp_comite | 61.7% +- 2.5 [57-66] | 78.5% +- 3.2 [73-83] | 74.4% +- 2.6 [66-79] | 12-15 | 11-14 |
| exp_segmentacion_v2 | 42.0% +- 5.6 [34-63] | 39.8% +- 7.1 [29-65] | 71.9% +- 6.3 [59-82] | 10-13 | 9-10 |
| exp_normalizacion | 49.0% +- 6.7 [37-60] | 65.0% +- 7.7 [50-80] | 64.2% +- 7.4 [49-78] | 12-14 | 8-9 |
| exp_medianas_ronda1 | 55.1% +- 2.7 [50-63] | 64.3% +- 5.9 [53-79] | 65.1% +- 2.8 [61-72] | 39-47 | 16-18 |

## D. Matrices de confusion en el informe (ajuste final)

**principal: etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 7 | 1 | 7 | 4 |
| tropical | 0 | 5 | 14 | 0 |
| carozo | 8 | 0 | 58 | 0 |

**principal: etapa 2 aislada (todas las manzanas del informe)**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 21 | 1 | 8 | 0 |
| comercial | 8 | 10 | 0 | 0 |
| descarte | 0 | 0 | 18 | 0 |

**original_reentrenado: etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 10 | 0 | 9 | 0 |
| tropical | 10 | 7 | 2 | 0 |
| carozo | 22 | 14 | 30 | 0 |

**original_reentrenado: etapa 2 aislada (todas las manzanas del informe)**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 25 | 0 | 5 | 0 |
| comercial | 8 | 7 | 3 | 0 |
| descarte | 6 | 0 | 12 | 0 |

**exp_medianas: etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 0 | 0 | 19 | 0 |
| tropical | 0 | 10 | 9 | 0 |
| carozo | 0 | 0 | 66 | 0 |

**exp_medianas: etapa 2 aislada (todas las manzanas del informe)**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 21 | 1 | 8 | 0 |
| comercial | 8 | 10 | 0 | 0 |
| descarte | 0 | 0 | 18 | 0 |

## F. Diagnostico basado unicamente en el fondo (marco del 5 % del borde, 5 bits)

Mirando SOLO el entorno de la foto (sin la fruta) se predice la clase con una tabla de consulta entrenada con 'desarrollo'. Si se acerca al sistema, parte de lo que este acierta puede ser el escenario.

| tarea | solo fondo: informe (ajuste final) | solo fondo: informe, 30 particiones internas | referencia mayoritaria | principal (ajuste final) |
|---|---|---|---|---|
| tipo | 59.6% (F1 macro 0.442, n=104) | 61.4% +- 3.1 | 63.5% | 67.3% |
| calidad (manzanas) | 77.3% (F1 macro 0.799, n=66) | 74.7% +- 5.6 | 45.5% | 74.2% |

Es un diagnostico, no una correccion: no hay fotos de la misma fruta en escenarios distintos, asi que no se puede separar el efecto del escenario del de la fruta. Ver tambien `scripts/robustez.py` (sustitucion de fondo e iluminacion).

## E. Comparacion pareada con la receta original reentrenada (informe, ajuste final)

- cadena total: diferencia media (principal - original reentrenado) = +19.2 puntos sobre 104 fotos; intervalo percentil 95 por remuestreo de grupos [-2.5, +42.3].
- tipo: diferencia media (principal - original reentrenado) = +22.1 puntos sobre 104 fotos; intervalo percentil 95 por remuestreo de grupos [+0.0, +42.4].

El intervalo describe la incertidumbre por la composicion del informe (pocos grupos); no se afirma significacion estadistica ni independencia: las fotos son de un mismo banco reutilizado y las rafagas pueden compartir fruta.

