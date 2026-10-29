# Diagnostico de confusion fondo-clase

Pregunta: si NO se mira la fruta sino solo un marco del 5% en el borde de la foto, cuanto se acierta? Si el entorno por si solo predice la clase, las etiquetas estan mezcladas con el escenario de la foto y la exactitud de cualquier sistema sobre este banco no prueba que mida la fruta.

## Entorno por origen (media del marco exterior)

| origen | fotos | fraccion de pixeles verdes del marco | saturacion media del marco | luminosidad L media |
|---|---|---|---|---|
| manzana_good | 99 | 6% | 0.25 | 137 |
| manzana_mixed | 60 | 97% | 0.61 | 141 |
| manzana_bad | 60 | 0% | 0.15 | 178 |
| naranja | 60 | 9% | 0.21 | 161 |
| banana | 60 | 12% | 0.22 | 171 |

## Exactitud en PRUEBA (media +- desvio entre 30 particiones por grupo)

| tarea | solo el entorno (5 bits, mediana de entrenamiento) | referencia mayoritaria | sistema mejorado (fruta) |
|---|---|---|---|
| tipo de fruta (etapa 1) | 67.2% +- 11.6 (F1 macro 0.532) | 63.8% +- 6.0 | 72.3% +- 8.9 (F1 macro 0.653) |
| calidad de manzana (etapa 2 aislada) | 77.6% +- 8.4 (F1 macro 0.792) | 44.3% +- 2.8 | 77.0% +- 11.4 (F1 macro 0.748) |

Lectura: si la columna 'solo el entorno' se acerca a la del sistema, parte de lo que el sistema 'aprende' puede ser el escenario (pasto, guantes, pisos) y no la fruta. Es un diagnostico, no una correccion: no hay forma de separar ambos efectos con este banco (no hay fotos de la misma fruta en escenarios distintos).

