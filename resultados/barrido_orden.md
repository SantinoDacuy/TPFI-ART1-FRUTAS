# Barrido de rho y sensibilidad al orden de presentacion

30 permutaciones aleatorias del entrenamiento + el orden original, para cada rho. Mezclar con una semilla fija permite repetir un orden, pero no elimina la dependencia del orden: ART1 es un algoritmo en linea y su resultado depende de la secuencia.

## Dataset entregado de morfologia (180 filas, una pasada)

Vectores distintos: 20; con etiquetas mezcladas: 9; **techo de pureza por vectores identicos: 84.44%** (ningun rho lo supera: vectores iguales con etiquetas distintas no se separan).

| rho | categorias media (min-max) | pureza online media (min-max) | pureza final media | corridas infructuosas |
|---|---|---|---|---|
| 0.3 | 4.6 (4-5) | 65.5% (56.7-72.8) | 66.5% | 0/31 |
| 0.35 | 6.0 (5-7) | 63.8% (53.9-71.1) | 66.4% | 0/31 |
| 0.4 | 6.0 (5-7) | 63.8% (53.9-71.1) | 66.4% | 0/31 |
| 0.45 | 6.5 (5-7) | 66.8% (59.4-73.9) | 70.3% | 0/31 |
| 0.5 | 6.5 (5-7) | 66.8% (59.4-73.9) | 70.3% | 0/31 |
| 0.55 | 9.4 (8-10) | 70.1% (62.8-77.8) | 73.0% | 0/31 |
| 0.6 | 9.4 (8-10) | 70.1% (62.8-77.8) | 73.0% | 0/31 |
| 0.65 | 9.5 (8-10) | 71.6% (66.1-77.2) | 74.2% | 0/31 |
| 0.7 | 14.5 (13-15) | 74.4% (69.4-81.1) | 79.1% | 0/31 |
| 0.75 | 14.5 (13-15) | 74.4% (69.4-81.1) | 79.1% | 0/31 |
| 0.8 | 18.0 (16-19) | 75.1% (70.0-80.6) | 80.6% | 0/31 |
| 0.85 | 18.9 (17-20) | 77.5% (73.3-81.7) | 83.9% | 0/31 |
| 0.9 | 18.9 (17-20) | 77.5% (73.3-81.7) | 83.9% | 0/31 |
| 0.95 | 18.9 (17-20) | 77.5% (73.3-81.7) | 83.9% | 0/31 |
| 1.0 | 18.9 (17-20) | 77.5% (73.3-81.7) | 83.9% | 0/31 |

## Dataset entregado de calidad (180 filas, una pasada)

Vectores distintos: 23; con etiquetas mezcladas: 10; **techo de pureza por vectores identicos: 86.67%** (ningun rho lo supera: vectores iguales con etiquetas distintas no se separan).

| rho | categorias media (min-max) | pureza online media (min-max) | pureza final media | corridas infructuosas |
|---|---|---|---|---|
| 0.3 | 6.5 (6-8) | 75.9% (68.9-82.2) | 76.4% | 0/31 |
| 0.35 | 8.0 (7-9) | 74.5% (63.3-80.0) | 76.8% | 0/31 |
| 0.4 | 8.0 (7-9) | 74.5% (63.3-80.0) | 76.8% | 0/31 |
| 0.45 | 8.2 (7-9) | 74.8% (63.3-79.4) | 77.0% | 0/31 |
| 0.5 | 8.2 (7-9) | 74.8% (63.3-79.4) | 77.0% | 0/31 |
| 0.55 | 12.6 (12-13) | 78.8% (73.9-83.9) | 81.6% | 0/31 |
| 0.6 | 12.6 (12-13) | 78.8% (73.9-83.9) | 81.6% | 0/31 |
| 0.65 | 12.9 (12-14) | 78.3% (73.3-83.3) | 81.4% | 0/31 |
| 0.7 | 17.3 (16-18) | 79.0% (74.4-83.3) | 84.6% | 0/31 |
| 0.75 | 17.3 (16-18) | 79.0% (74.4-83.3) | 84.6% | 0/31 |
| 0.8 | 20.2 (18-22) | 81.0% (77.2-84.4) | 86.4% | 0/31 |
| 0.85 | 20.8 (19-23) | 80.6% (74.4-83.9) | 85.8% | 0/31 |
| 0.9 | 20.8 (19-23) | 80.6% (74.4-83.9) | 85.8% | 0/31 |
| 0.95 | 20.8 (19-23) | 80.6% (74.4-83.9) | 85.8% | 0/31 |
| 1.0 | 20.8 (19-23) | 80.6% (74.4-83.9) | 85.8% | 0/31 |

## Version mejorada - morfologia (entrenamiento de la particion principal, 190 fotos, hasta 10 pasadas)

Vectores distintos: 69; con etiquetas mezcladas: 9; techo de pureza por vectores identicos: 94.21%.

| rho | categorias media (min-max) | pureza online media | pureza final media | F1 macro validacion media (min-max) | infructuosas |
|---|---|---|---|---|---|
| 0.3 | 11.4 (9-13) | 68.3% | 73.1% | 0.487 (0.240-0.742) | 0/31 |
| 0.35 | 14.3 (12-16) | 69.1% | 73.0% | 0.483 (0.262-0.760) | 0/31 |
| 0.4 | 14.3 (12-16) | 69.1% | 73.0% | 0.483 (0.262-0.760) | 0/31 |
| 0.45 | 17.3 (15-20) | 70.6% | 72.9% | 0.471 (0.244-0.751) | 0/31 |
| 0.5 | 17.3 (15-20) | 70.6% | 72.9% | 0.471 (0.244-0.751) | 0/31 |
| 0.55 | 25.2 (23-27) | 73.8% | 81.8% | 0.619 (0.369-0.721) | 0/31 |
| 0.6 | 25.8 (23-28) | 73.7% | 82.7% | 0.638 (0.444-0.721) | 0/31 |
| 0.65 | 30.3 (28-34) | 77.2% | 84.7% | 0.629 (0.314-0.743) | 0/31 |
| 0.7 | 38.9 (37-42) | 81.7% | 88.8% | 0.665 (0.602-0.734) | 0/31 |
| 0.75 | 39.7 (37-42) | 81.7% | 89.1% | 0.680 (0.619-0.746) | 0/31 |
| 0.8 | - (corrida infructuosa) | - | - | - (corrida infructuosa) | 31/31 |
| 0.85 | - (corrida infructuosa) | - | - | - (corrida infructuosa) | 31/31 |
| 0.9 | - (corrida infructuosa) | - | - | - (corrida infructuosa) | 31/31 |
| 0.95 | - (corrida infructuosa) | - | - | - (corrida infructuosa) | 31/31 |
| 1.0 | - (corrida infructuosa) | - | - | - (corrida infructuosa) | 31/31 |

## Version mejorada - calidad (entrenamiento de la particion principal, 122 fotos, hasta 10 pasadas)

Vectores distintos: 33; con etiquetas mezcladas: 11; techo de pureza por vectores identicos: 84.43%.

| rho | categorias media (min-max) | pureza online media | pureza final media | F1 macro validacion media (min-max) | infructuosas |
|---|---|---|---|---|---|
| 0.3 | 8.3 (8-9) | 68.2% | 70.7% | 0.706 (0.535-0.844) | 0/31 |
| 0.35 | 9.4 (9-11) | 65.4% | 68.8% | 0.700 (0.517-0.837) | 0/31 |
| 0.4 | 9.4 (9-11) | 65.4% | 68.8% | 0.700 (0.517-0.837) | 0/31 |
| 0.45 | 11.0 (10-13) | 67.5% | 72.2% | 0.736 (0.536-0.837) | 0/31 |
| 0.5 | 11.0 (10-13) | 67.5% | 72.2% | 0.736 (0.536-0.837) | 0/31 |
| 0.55 | 15.8 (15-17) | 72.3% | 76.9% | 0.613 (0.560-0.768) | 0/31 |
| 0.6 | 15.8 (15-17) | 72.3% | 76.9% | 0.613 (0.560-0.768) | 0/31 |
| 0.65 | 18.0 (17-19) | 73.0% | 78.5% | 0.643 (0.569-0.790) | 0/31 |
| 0.7 | 22.4 (22-23) | 74.8% | 81.3% | 0.644 (0.622-0.686) | 0/31 |
| 0.75 | 22.4 (22-23) | 74.8% | 81.3% | 0.644 (0.622-0.686) | 0/31 |
| 0.8 | 29.0 (29-29) | 77.1% | 82.8% | 0.649 (0.631-0.651) | 0/31 |
| 0.85 | 33.0 (33-33) | 77.9% | 84.4% | 0.651 (0.651-0.651) | 0/31 |
| 0.9 | 33.0 (33-33) | 77.9% | 84.4% | 0.651 (0.651-0.651) | 0/31 |
| 0.95 | 33.0 (33-33) | 77.9% | 84.4% | 0.651 (0.651-0.651) | 0/31 |
| 1.0 | 33.0 (33-33) | 77.9% | 84.4% | 0.651 (0.651-0.651) | 0/31 |

## Sensibilidad al orden de las metricas de punta a punta (misma particion, rho fijo)

| sistema | particion | cadena total media +- sd (min-max) | tipo F1 macro media +- sd | categorias (morf/cal) media (min-max) |
|---|---|---|---|---|
| original reentrenado | validacion | 44.2% +- 5.0 (31.0-53.5) | 0.572 +- 0.043 | 17.6 (15-20) / 15.7 (14-17) |
| original reentrenado | prueba | 39.7% +- 4.5 (32.1-52.6) | 0.518 +- 0.035 | 17.6 (15-20) / 15.7 (14-17) |
| mejorado | validacion | 60.6% +- 5.5 (50.7-70.4) | 0.666 +- 0.058 | 25.5 (23-28) / 11.1 (10-13) |
| mejorado | prueba | 50.8% +- 7.1 (38.5-69.2) | 0.529 +- 0.048 | 25.5 (23-28) / 11.1 (10-13) |
