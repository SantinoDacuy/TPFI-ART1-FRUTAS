# Linea de base reproducible (primer avance tal como fue entregado)

Codigo, datasets y modelos: `baseline_original/` (snapshot del commit `f01b4dd`).  
Python 3.14.0 - Windows-11-10.0.26200-SP0.

## A. Datasets

| etapa | filas | variables | vectores distintos | vectores con etiquetas mezcladas | techo de pureza | vectores nulos | columnas constantes |
|---|---|---|---|---|---|---|---|
| morfologia | 180 | 6 | 20 | 9 (100 muestras) | 84.44% | 0 | tiene_tallo |
| calidad | 180 | 5 | 23 | 10 (107 muestras) | 86.67% | 16 | - |

*Techo de pureza*: maxima pureza alcanzable por CUALQUIER agrupamiento de estos vectores (vectores identicos con etiquetas distintas no se separan variando rho).

## B. Corridas de referencia (una pasada, orden original)

| etapa | rho | clusters | pureza online | esperado | coincide |
|---|---|---|---|---|---|
| morfologia | 0.7 | 14 | 76.11% | 14 / 76.11% | si |
| morfologia | 0.85 | 19 | 75.56% | 19 / 75.56% | si |
| calidad | 0.7 | 17 | 81.11% | 17 / 81.11% | si |
| calidad | 0.85 | 21 | 81.67% | 21 / 81.67% | si |
| calidad | 0.95 | 21 | 81.67% | 21 / 81.67% | si |

## C. Evaluacion fotografica con los modelos entregados (reinferencia: incluye fotos de entrenamiento)

| medida | obtenido | esperado | coincide |
|---|---|---|---|
| Tipo correcto (de 339) | 167 | 167 | si |
| Manzanas derivadas a calidad (de 219) | 96 | 96 | si |
| Tipo y calidad correctos (de 219) | 65 | 65 | si |
| Naranjas/bananas derivadas a calidad (de 120) | 44 | 44 | si |

Exactitud de tipo 49.3%, F1 macro 0.479; referencia 'siempre carozo': 64.6% / F1 macro 0.262. Sin resonancia en etapa 1: 0.

| estrato | fotos | tipo correcto | derivadas a calidad |
|---|---|---|---|
| banana | 60 | 51 | 4 |
| manzana_bad | 60 | 24 | 24 |
| manzana_good | 99 | 61 | 61 |
| manzana_mixed | 60 | 11 | 11 |
| naranja | 60 | 20 | 40 |

## D. Otros hallazgos comprobados

- **Mapeo online vs prototipos finales (morfologia, rho=0.85)**: 0/180 muestras de entrenamiento ya no resuenan con los prototipos finales; 75 cambian de categoria; 5/19 categorias cambiarian de etiqueta o quedarian vacias. El modelo entregado coincide con reentrenar su CSV con su rho.
- **Mapeo online vs prototipos finales (calidad, rho=0.7)**: 0/180 muestras de entrenamiento ya no resuenan con los prototipos finales; 44 cambian de categoria; 2/17 categorias cambiarian de etiqueta o quedarian vacias. El modelo entregado coincide con reentrenar su CSV con su rho.
- **Valor invalido pasa a metadato**: caracteristicas detectadas ['a', 'b'], metadatos ['c', 'tipo_real'] (la columna c tenia un '2' y paso a metadato; la red se entreno con 2 variables).
- **Corrida infructuosa**: `CarGross.py --rho 0.95 --max-categorias 5` termino con codigo de salida **0** (salida: ['CORRIDA INFRUCTUOSA (rho=0.95): Se alcanzo el limite de 5 categorias sin resonancia. Esto indica proliferacion descontrolada: rho=0.95 es demasiado alto para el ruido de los datos. Sugerencia: bajar rho.', 'Se procesaron 13/180 registros antes de cortar.']).
- **rho en rutas de regeneracion**: modelo entregado de morfologia con rho=0.85; lineas que usan otro valor:
  - `probar_imagen.py:48: def asegurar_modelos(rho_morf=0.75, rho_cal=0.70):`
  - `probar_imagen.py:56: red_morf, meta_morf = entrenar_modelo(RUTA_CSV_MORF, rho=rho_morf)`
  - `probar_imagen.py:63: red_cal, meta_cal = entrenar_modelo(RUTA_CSV_CAL, rho=rho_cal)`
  - `scripts/curar_dataset.py:31: from CarGross import entrenar_modelo`
  - `scripts/curar_dataset.py:342: red_morf, meta_morf = entrenar_modelo(RUTA_CSV_MORF, rho=0.75)`
  - `scripts/curar_dataset.py:346: red_cal, meta_cal = entrenar_modelo(RUTA_CSV_CAL, rho=0.70)`
