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
| principal | PRINCIPAL compatible con el primer avance | 0.65 / 0.35 | 62/104 (59.6%) IC [40.7%, 76.6%] | 49/66 (74.2%) | 79/104 (76.0%) [57.9%, 90.7%] | 0.615 | 74.2% / 0.740 | 49/66 (74.2%) | 66/66 (100.0%) | 24/38 (63.2%) | 0 / 0 | 11 (1) / 10 (1) |
| exp_sin_tope | experimental: sin el tope de 14 categorias del primer avance | 0.7 / 0.55 | 68/104 (65.4%) IC [47.1%, 81.2%] | 55/66 (83.3%) | 78/104 (75.0%) [56.8%, 89.4%] | 0.611 | 84.8% / 0.845 | 55/65 (84.6%) | 65/66 (98.5%) | 25/38 (65.8%) | 0 / 0 | 14 (1) / 16 (2) |
| exp_medianas | experimental: umbrales morfologicos por mediana | 0.3 / 0.35 | 59/104 (56.7%) IC [37.5%, 72.8%] | 49/66 (74.2%) | 75/104 (72.1%) [56.0%, 84.6%] | 0.504 | 74.2% / 0.740 | 49/65 (75.4%) | 65/66 (98.5%) | 27/38 (71.1%) | 2 / 0 | 6 (0) / 10 (1) |
| exp_protuberancia | experimental: tiene_tallo por protuberancia (sin anotaciones) | 0.6 / 0.35 | 58/104 (55.8%) IC [36.8%, 73.5%] | 48/66 (72.7%) | 75/104 (72.1%) [53.6%, 88.0%] | 0.525 | 74.2% / 0.740 | 48/65 (73.8%) | 65/66 (98.5%) | 27/38 (71.1%) | 0 / 0 | 10 (0) / 10 (1) |
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
| principal | morfologia | 11 | 1 | 0 | 17.0 | 51 | [51, 36, 28, 27, 25, 17, 15, 14, 12, 9, 1] |
| principal | calidad | 10 | 1 | 0 | 13.0 | 36 | [36, 29, 25, 24, 16, 10, 7, 3, 2, 1] |
| exp_sin_tope | morfologia | 14 | 1 | 0 | 14.0 | 39 | [39, 36, 33, 25, 25, 22, 15, 13, 9, 8, 4, 3, 2, 1] |
| exp_sin_tope | calidad | 16 | 2 | 0 | 7.0 | 27 | [27, 23, 19, 19, 15, 13, 10, 9, 5, 3, 2, 2, 2, 2, 1, 1] |
| exp_medianas | morfologia | 6 | 0 | 0 | 40.0 | 76 | [76, 61, 58, 22, 9, 9] |
| exp_medianas | calidad | 10 | 1 | 0 | 13.0 | 36 | [36, 29, 25, 24, 16, 10, 7, 3, 2, 1] |
| exp_protuberancia | morfologia | 10 | 0 | 0 | 14.5 | 65 | [65, 62, 33, 22, 20, 9, 7, 7, 6, 4] |
| exp_protuberancia | calidad | 10 | 1 | 0 | 13.0 | 36 | [36, 29, 25, 24, 16, 10, 7, 3, 2, 1] |
| exp_medianas_ronda1 | morfologia | 42 | 13 | 0 | 2.5 | 30 | [30, 21, 19, 18, 14, 12, 12, 11, 11, 10, 8, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| exp_medianas_ronda1 | calidad | 17 | 2 | 0 | 7.0 | 30 | [30, 19, 18, 16, 13, 10, 10, 9, 7, 5, 5, 3, 2, 2, 2, 1, 1] |

## B. Variabilidad entre 30 particiones internas (media +- desvio [minimo-maximo])

Entrenamiento/validacion salen de 'desarrollo'; el informe es el mismo bloque en todas. Cada particion entrena con ~70 % de desarrollo.

| variante | validacion: cadena total | validacion: cadena manzanas | informe: cadena total | informe: cadena manzanas | informe: tipo exactitud | informe: tipo F1 macro | informe: otras frutas derivadas (n) | categorias (morf/cal) | rho elegidos (morf/cal) |
|---|---|---|---|---|---|---|---|---|---|
| original_reentrenado | 41.2% +- 7.8 [27-59] | 35.0% +- 10.0 [18-49] | 34.2% +- 5.2 [21-44] | 33.8% +- 9.6 [17-48] | 48.1% +- 8.2 [35-62] | 0.410 +- 0.058 | 18.3 +- 5.9 | 15.2 / 14.1 | 0.85 / 0.70 |
| compatible_minima | 51.8% +- 8.6 [38-67] | 63.8% +- 8.7 [48-84] | 49.7% +- 3.7 [43-59] | 61.8% +- 9.6 [45-77] | 65.3% +- 4.4 [58-71] | 0.495 +- 0.054 | 23.1 +- 6.5 | 10.0 / 8.5 | 0.68 / 0.44 |
| principal | 62.4% +- 6.9 [48-74] | 65.7% +- 7.7 [51-86] | 60.0% +- 3.6 [53-68] | 73.0% +- 6.6 [58-85] | 74.0% +- 2.7 [66-77] | 0.598 +- 0.072 | 22.6 +- 4.7 | 9.8 / 9.6 | 0.62 / 0.40 |
| exp_sin_tope | 64.4% +- 6.9 [48-77] | 67.6% +- 8.1 [55-86] | 61.3% +- 3.1 [55-68] | 73.6% +- 5.7 [58-85] | 74.5% +- 2.2 [69-77] | 0.619 +- 0.059 | 21.4 +- 4.1 | 13.0 / 15.1 | 0.67 / 0.55 |
| exp_medianas | 53.4% +- 10.5 [38-74] | 69.3% +- 8.3 [47-88] | 56.9% +- 4.6 [48-66] | 72.1% +- 6.0 [61-83] | 70.7% +- 4.2 [61-76] | 0.509 +- 0.078 | 25.0 +- 4.9 | 8.8 / 9.6 | 0.34 / 0.40 |
| exp_protuberancia | 54.1% +- 8.4 [42-74] | 66.3% +- 10.3 [48-92] | 57.2% +- 5.6 [48-69] | 71.3% +- 6.9 [58-85] | 70.8% +- 3.9 [63-81] | 0.518 +- 0.117 | 23.3 +- 7.5 | 10.0 / 9.6 | 0.56 / 0.40 |
| exp_medianas_ronda1 | 65.9% +- 8.2 [47-80] | 68.7% +- 9.5 [50-86] | 57.1% +- 5.6 [45-66] | 63.0% +- 9.1 [39-74] | 68.5% +- 5.6 [56-79] | 0.605 +- 0.070 | 18.1 +- 5.0 | 34.3 / 14.1 | 0.72 / 0.52 |

Fallos (corridas infructuosas) por variante: original_reentrenado: 0, compatible_minima: 0, principal: 0, exp_sin_tope: 0, exp_medianas: 0, exp_protuberancia: 0, exp_medianas_ronda1: 0.

## C. Sensibilidad al orden de presentacion (ajuste final, 30 ordenes aleatorios, informe)

| variante | cadena total: media +- desvio [min-max] | cadena de manzanas | tipo exactitud | categorias morf (min-max) | categorias cal (min-max) |
|---|---|---|---|---|---|
| original_reentrenado | 31.6% +- 3.3 [27-40] | 23.5% +- 6.9 [11-47] | 41.5% +- 4.7 [34-53] | 15-18 | 14-18 |
| compatible_minima | 47.0% +- 5.9 [36-56] | 60.7% +- 10.1 [42-74] | 63.2% +- 7.6 [47-72] | 9-13 | 7-9 |
| principal | 54.9% +- 4.0 [49-63] | 70.4% +- 4.7 [59-80] | 72.5% +- 2.7 [67-77] | 10-11 | 10-11 |
| exp_sin_tope | 63.4% +- 2.8 [56-67] | 78.5% +- 2.8 [71-83] | 74.6% +- 2.6 [67-77] | 13-15 | 16-17 |
| exp_medianas | 54.0% +- 3.3 [43-61] | 69.6% +- 5.1 [56-80] | 70.5% +- 3.4 [56-73] | 6-8 | 10-11 |
| exp_protuberancia | 55.0% +- 4.7 [44-65] | 66.7% +- 4.7 [56-74] | 72.0% +- 3.8 [61-78] | 10-11 | 10-11 |
| exp_medianas_ronda1 | 55.1% +- 2.7 [50-63] | 64.3% +- 5.9 [53-79] | 65.1% +- 2.8 [61-72] | 39-47 | 16-18 |

## D. Matrices de confusion en el informe (ajuste final)

**principal: etapa 1 (tipo)**

| real \ predicha | citrico | tropical | carozo | (sin etiqueta) |
|---|---|---|---|---|
| citrico | 7 | 1 | 11 | 0 |
| tropical | 0 | 6 | 13 | 0 |
| carozo | 0 | 0 | 66 | 0 |

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
| citrico | 0 | 0 | 18 | 1 |
| tropical | 0 | 10 | 9 | 0 |
| carozo | 0 | 0 | 65 | 1 |

**exp_medianas: etapa 2 aislada (todas las manzanas del informe)**

| real \ predicha | premium | comercial | descarte | (sin etiqueta) |
|---|---|---|---|---|
| premium | 21 | 1 | 8 | 0 |
| comercial | 8 | 10 | 0 | 0 |
| descarte | 0 | 0 | 18 | 0 |

## E. Comparacion pareada con la receta original reentrenada (informe, ajuste final)

- cadena total: diferencia media (principal - original reentrenado) = +26.0 puntos sobre 104 fotos; intervalo percentil 95 por remuestreo de grupos [+4.2, +46.8].
- tipo: diferencia media (principal - original reentrenado) = +30.8 puntos sobre 104 fotos; intervalo percentil 95 por remuestreo de grupos [+11.2, +48.1].

El intervalo describe la incertidumbre por la composicion del informe (pocos grupos); no se afirma significacion estadistica ni independencia: las fotos son de un mismo banco reutilizado y las rafagas pueden compartir fruta.

