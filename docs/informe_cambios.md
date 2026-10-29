# Informe de cambios, limitaciones y pendientes

Material para el manual y el informe final. Cada afirmación indica de dónde sale (archivo o comando). Entorno de las
cifras: Python 3.14.0, OpenCV 5.0.0.93, NumPy 2.4.6 (`resultados/verificacion/`).

**Estructura.** *Parte A* = estado final (segunda ronda, protocolo con bloque de informe reservado). *Parte B* = primera
ronda (protocolo exploratorio previo; se conserva como historia y respaldo, sus cifras **no** deben citarse como
desempeño final). El primer avance no se tocó: `datasets/`, `modelos/`, `ejemplos/`, `preprocesador_calidad.py`, las
funciones v1 de `preprocesador.py` y el snapshot `baseline_original/` (commit `f01b4dd`). No se hicieron commits ni se
eliminó ningún respaldo (la primera ronda se movió con copia verificada por hash a `ronda1_medianas/`).

---

# PARTE A — Estado final (segunda ronda)

## A1. Resumen

* **Recomendación de configuración principal** (compatible con el primer avance): `modelos_v2/principal/`,
  `datasets_v2/principal/`, parámetros en `config/parametros_etapas.csv`. Conserva ART1, las dos etapas, la entrada CSV,
  las variables comprometidas con sus **cortes originales** (aspecto 0,80; rugosidad 150; solidez 0,90; gris 130; desvío
  de tono 12; `tiene_tallo` por defectos de convexidad con corte 0,06), la calidad con medianas de lote (como ya hacía el
  primer avance) y el **criterio de éxito del primer avance** (menos de 15 categorías para 3 clases). Agrega dos
  descriptores cromáticos en morfología (`croma_alto`, `naranja_dominante`) y uno en calidad (`croma_alto`).
* **Desempeño demostrable** (bloque de informe de 104 fotos, 66 manzanas; 30 particiones internas):
  cadena completa **63,0 % ± 4,3** contra **34,2 % ± 5,2** de la receta original reentrenada; cadena de manzanas
  72,5 % ± 7,3 contra 33,8 % ± 9,6; referencia "siempre manzana + calidad mayoritaria": 45,5 % en manzanas. En el **ajuste
  final** (una sola evaluación) la principal acierta 52,9 % [IC95 33,0–71,2] contra 33,7 % [22,1–50,0] y la diferencia
  pareada tiene intervalo [−2,5; +42,3] puntos: **incluye el cero**. No se afirma significación estadística.
* **El resultado depende de una decisión que el equipo debe validar**: el criterio de < 15 categorías cuesta unos 5 a 6
  puntos de cadena (68,4 % ± 4,0 sin tope, con 31–45 categorías); se respetó porque es una definición del primer avance.
* **Lo que no funcionó** (resultado negativo conservado): segmentación v2, normalización de iluminación, comité de 11
  redes, derivación condicionada a la coincidencia, soporte mínimo de categoría, medianas morfológicas, detector de tallo
  por protuberancia, y todos los demás descriptores cromáticos probados (§A5).
* **Límites que siguen abiertos**: banco ya inspeccionado y filtrado; sin id de fruta física ni de sesión; **calidad
  confundida con el escenario** (mirando solo el borde de la foto: 74,7 % ± 5,6 en 30 particiones contra 74,2 % del
  sistema en el ajuste final); **las predicciones no son estables** ante el fondo ni la iluminación (§A7); `tiene_tallo` sin
  anotaciones; separación cítrico/manzana débil (otras frutas derivadas a calidad: 21 de 38).

## A2. Punto 1: guardado desde CSV v2 (corregido)

**Falla verificada**: `CarGross.py --csv datasets_v2/calidad_entrenamiento.csv --guardar-modelo X` guardaba 6 variables
pero adjuntaba el preprocesamiento v1 de 5; `cargar_modelo` lo aceptaba.

**Corrección** (`CarGross.py`, `validacion.py`, `extraccion.py`):

* El preprocesamiento exacto viaja como **CSV documentado** (`preprocesamiento_<etapa>.csv`: filas `global` con versión,
  etapa, segmentador, tamaño de canvas, normalización, opciones y origen de los umbrales; filas `variable` con medida
  continua, operador y umbral exacto, en el orden del vector). `--guardar-modelo` lo recupera junto al dataset o lo recibe
  con `--preprocesamiento`, y lo **valida antes de entrenar o escribir** (`validar_preprocesamiento_dict`): versión,
  etapa, segmentador, canvas, orden canónico de variables, una regla con medida existente y operador válido y un umbral
  numérico finito por variable, ninguna regla sobrante, coincidencia exacta (nombres y orden) con las columnas del CSV.
* **Tipo de modelo**: `fotografias` (preprocesamiento completo) o `vectores` (guardado sin preprocesamiento: clasifica
  filas de un CSV pero `probar_imagen.py`/`inferencia.py` lo rechazan con un mensaje). `RedART1.guardar` valida también el
  modelo completo; `cargar_modelo` repite la validación. Los modelos de formato 1 siguen cargando en modo legado.
* **Nunca se supone v1**: sin sidecar ni declaración se guarda un modelo de vectores; `--preprocesamiento legado` solo
  se acepta si el CSV trae exactamente las variables del primer avance.
* Pruebas (`tests/test_guardado_csv_v2.py`, ~20 pruebas): entrenar desde CSV v2 → guardar → recargar → procesar una foto
  con la **misma representación** (vector y umbrales idénticos); predicciones idénticas antes y después de guardar; 6
  preprocesamientos defectuosos (variable o regla ausente, medida u operador desconocidos, umbral no numérico, fila
  repetida, etapa equivocada) más orden distinto y preprocesamiento de otra etapa, todos rechazados **sin escribir el modelo**;
  `MEDIDAS_CONOCIDAS` coincide con lo que produce `medir_foto`. `tests/test_modelos_v2.py` comprueba que cada modelo
  entregado se **reproduce desde su CSV con la CLI** (mismos pesos, etiquetas y preprocesamiento).

## A3. Punto 2: variante principal compatible y variantes experimentales separadas

| variante | carpeta | qué es |
|---|---|---|
| **principal** | `modelos_v2/principal/`, `datasets_v2/principal/` | definiciones y cortes originales; ≤ 14 categorías; compatible con el primer avance |
| alternativa orden canónico | `…/alternativa_orden_canonico/` | igual, con orden de presentación determinista (mismo resultado siempre) |
| experimental medianas | `…/experimental_medianas/` | cambia el significado de `es_esferica` y demás (corte = mediana del entrenamiento) |
| experimental segmentación v2 | `…/experimental_segmentacion_v2/` | segmentación por candidatos |
| experimental normalización | `…/experimental_normalizacion/` | estiramiento de contraste antes de medir |
| ronda 1 (medianas) | `…/ronda1_medianas/` | respaldo de la primera ronda |

* Se **reentrenó y evaluó la combinación final con los cortes originales** (no se estimó restando efectos de la
  ablación de la primera ronda ni se atribuyó el 60,2 % de la receta de medianas). La receta de medianas de la primera
  ronda, reevaluada en este protocolo: 57,1 % ± 5,6 con 34 categorías de morfología (`exp_medianas_ronda1`).
* **`tiene_tallo`**: se **corrigió la incompatibilidad de OpenCV 5** (`convexityDefects` devuelve (N,4); el código v1
  exigía 3-D y dejaba la variable en 0) conservando el criterio y el corte originales (`profundidad_defecto_v2`). El
  detector por protuberancia es **otro detector**, no validado (no hay anotaciones de referencia): variante experimental
  (−0,028 ± 0,013 en validación).
* `desvio_hue` lineal → desvío circular: corrección de una medición (los rojos cruzan el cero del tono); conserva el
  corte 12 y la interpretación. Efecto en validación: +0,018 ± 0,008.

## A4. Protocolo de evaluación (punto 4)

* **Bloque de informe reservado**: 30 % de los **grupos** de fotos (≤ 3 s entre tomas; proxy de ráfaga porque no hay id de
  fruta física), estratificado por origen, semilla 8 elegida solo por el equilibrio de tamaños: 104 fotos (66 manzanas,
  38 otras frutas). Resto = 235 fotos de desarrollo.
* **Selección**: 30 particiones internas entrenamiento/validación (70/30 de los grupos de desarrollo). Umbrales solo con
  el entrenamiento de cada partición; ρ en validación; variantes por la regla declarada en `scripts/busqueda.py`
  (obligatoria / corrección no inferior a −0,01 / mejora ≥ +0,01 y > 1 error estándar; métrica: cadena completa en
  validación, diferencias pareadas). El informe **no interviene en ninguna decisión** (pruebas: umbrales, pesos y ρ no
  cambian al alterar las fotos del informe).
* **Honestidad sobre la reutilización**: las 339 fotos ya se habían inspeccionado y usado en la primera ronda; esta
  partición **no es un conjunto externo intacto**. Además, durante la búsqueda se vieron las cifras internas sobre el
  informe (columnas `prueba_*` de `busqueda.csv`) y la búsqueda se repitió tres veces al agregar candidatos
  (`resultados/ronda2/historial_busquedas/`): la idea de los descriptores `*_dominante` y del orden canónico surgió tras
  ver errores; las decisiones se tomaron con validación.
* **Denominadores** (docstring de `scripts/comparar_ronda2.py`): N = fotos del informe; cadena total = (manzanas con tipo
  y calidad correctos + otras frutas con tipo correcto)/N; cadena de manzanas = correctas/M; calidad aislada = sobre las
  M manzanas; calidad acoplada = sobre las manzanas derivadas; novedad o categoría sin etiqueta = error, informada aparte.
* **Dispersión**: "± x" es el desvío entre particiones que comparten fotos (no un intervalo independiente); "EE" = error
  estándar de la diferencia pareada entre particiones (descriptivo); "IC95" = remuestreo de grupos del informe
  (2000 repeticiones, pocos grupos: ancho). Ninguno prueba significación ni independencia.

## A5. Búsqueda de mejoras (punto 3) — `resultados/ronda2/busqueda.csv`

Δ = cambio de la cadena en validación (media entre 30 particiones ± EE); "inf" = cambio en el informe (solo informativo).

| paso | decisión | Δ validación | inf |
|---|---|---|---|
| P1 etiquetado con prototipos finales | obligatoria | −0,000 ± 0,012 | +0,033 |
| P2 etapa 1 con manzanas de todas las calidades | obligatoria | **+0,062 ± 0,010** | +0,053 |
| P3 ρ elegido en validación | obligatoria | +0,043 ± 0,009 | +0,027 |
| P4 umbrales de calidad solo con entrenamiento | obligatoria | −0,010 ± 0,006 | +0,029 |
| P5 `tiene_tallo` = mismo criterio corregido | corrección | −0,009 ± 0,007 | −0,018 |
| P6 desvío de tono circular | corrección | +0,018 ± 0,008 | +0,034 |
| P7 cortes originales de morfología | obligatoria | +0,009 ± 0,005 | +0,004 |
| P8 tope de 14 categorías (criterio del primer avance) | obligatoria | −0,006 ± 0,003 | −0,008 |
| N1 normalización de iluminación (3 opciones) | **no adoptada** | mejor: −0,049 ± 0,015 | −0,066 |
| X morfología 1.ª: `croma_alto` | adoptada | **+0,056 ± 0,012** | +0,059 |
| X morfología 2.ª: `naranja_dominante` | adoptada | **+0,037 ± 0,012** | +0,032 |
| X morfología 3.ª (9 candidatos) | ninguna | mejor: −0,031 ± 0,013 | |
| X calidad 1.ª: `croma_alto` | adoptada | +0,014 ± 0,004 | +0,030 |
| X calidad 2.ª (`sin_oscurecimiento` y otros) | ninguna | mejor: +0,006 ± 0,005 | −0,000 |
| M1 desempate global | no adoptada | +0,002 ± 0,002 | +0,005 |
| M3 color del descriptor: núcleo / balance de blancos | no adoptada | mejor: −0,021 ± 0,015 | |
| M4 orden de presentación: **menos activos primero** | adoptada | **+0,026 ± 0,011** | +0,012 |
| M4 orden canónico (determinista) | no adoptada | +0,004 ± 0,016 | +0,007 |
| M2 hasta estabilizar (10 pasadas) | no adoptada | −0,007 ± 0,003 | +0,006 |
| M5 ρ de la etapa 1 por la cadena | no adoptada | +0,002 ± 0,002 | +0,004 |
| M6 soporte mínimo 2 / 3 muestras | no adoptada | −0,004 / −0,009 | |
| M7 derivar solo con coincidencia ≥ 0,75 / = 1 | no adoptada | **−0,254 / −0,390** | −0,305 |
| M8 segmentación v2 | no adoptada | −0,100 ± 0,020 | −0,149 |
| *E1 tallo por protuberancia* (exp.) | no adoptable | −0,028 ± 0,013 | −0,013 |
| *E2 medianas morfológicas* (exp.) | no adoptable | −0,085 ± 0,016 | −0,036 |
| *E3 sin tope de categorías* (exp.) | no adoptable | **+0,062 ± 0,009** | +0,054 |
| *E4 comité de 11 redes* (exp.) | no adoptable | −0,064 ± 0,015 | −0,020 |

Lecturas:

* **Cobertura** (P2) y **ρ en validación** (P3) son lo que más mejora entre los cambios de método.
* **Color**: `croma_alto` (croma de Lab, poco sensible a la luminosidad) y `naranja_dominante` (fracción de píxeles de
  tono naranja) son los descriptores adicionales útiles; el gradiente de color (el ejemplo que mencionó la docente), el
  tono medio, la saturación, el rojo/amarillo/verde dominante **empeoraron** la validación y no se adoptan.
  Comprobación individual: `busqueda.csv`, filas `Xmor*` y `Xcal*`.
* **Orden de presentación**: presentar primero los patrones con menos bits activos mejora (+0,026, 1,9 EE); presentar
  primero los más activos destruye el resultado (−0,27). El orden canónico determinista mejora +0,004 ± 0,016 frente al
  orden aleatorio contra +0,026 ± 0,011 de "menos activos primero": la diferencia entre ambos (≈ 0,02) **no es concluyente**
  con estos errores (y en el informe dan 62,5 % y 63,0 %); por eso el canónico se entrega como alternativa (§A8).
* **Fragmentación**: el tope de 14 categorías y el soporte mínimo no ayudan; el comité no mejora y cuesta 11×.
* **Costo de las mejoras**: los descriptores añaden 2 bits a morfología y 1 a calidad; el cómputo es despreciable
  (medir las 339 fotos con segmentación v1 lleva segundos en paralelo). La segmentación v2 es ~7–10 veces más lenta
  (30–60 s las 339 fotos) y no aportó; el comité cuesta 11 redes.

## A6. Comparación de variantes — `resultados/ronda2/comparacion.md`

Ajuste final (todo `desarrollo`, ρ = mediana de los elegidos, ≤ 14 categorías) evaluado **una vez** en el informe (N = 104,
M = 66, otras frutas = 38). Referencias mayoritarias: tipo 63,5 % (F1 macro 0,259); cadena de manzanas 45,5 %.

| variante | cadena total [IC95] | cadena manzanas | tipo exactitud / F1 macro | calidad aislada / acoplada | derivadas: manzanas / otras | categorías morf / cal (unitarias) |
|---|---|---|---|---|---|---|
| original entregado (histórico, optimista*) | 42,3 % [28,6–53,5] | 31,8 % | 52,9 % / 0,509 | 68,2 % / 65,6 % | 32/66 · 10/38 | 19 / 17 |
| original reentrenado | 33,7 % [22,1–50,0] | 27,3 % | 45,2 % / 0,413 | 66,7 % / 60,0 % | 30/66 · 11/38 | 16 (0) / 17 (1) |
| compatible mínima (sin descriptores) | 51,9 % [35,8–66,4] | 69,7 % | 71,2 % / 0,469 | 69,7 % / 69,7 % | 66/66 · 30/38 | 10 (1) / 8 (1) |
| **principal** | **52,9 % [33,0–71,2]** | 65,2 % | 67,3 % / 0,537 | 74,2 % / 74,1 % | 58/66 · 21/38 | 11 (1) / 11 (2) |
| alternativa orden canónico | 60,6 % [40,9–77,9] | 78,8 % | 72,1 % / 0,555 | 80,3 % / 81,2 % | 64/66 · 26/38 | 11 (0) / 10 (1) |
| exp. sin tope de categorías | 72,1 % [55,2–86,1] | 78,8 % | 79,8 % / 0,731 | 84,8 % / 86,7 % | 60/66 · 15/38 | 45 (12) / 16 (5) |
| exp. medianas | 56,7 % [37,6–72,6] | 74,2 % | 73,1 % / 0,505 | 74,2 % / 74,2 % | 66/66 · 28/38 | 9 (0) / 11 (2) |
| exp. medianas (receta ronda 1) | 54,8 % [42,6–69,0] | 59,1 % | 66,3 % / 0,596 | 80,3 % / 76,5 % | 51/66 · 13/38 | 42 (13) / 17 (2) |
| exp. tallo por protuberancia | 57,7 % [38,9–74,8] | 68,2 % | 72,1 % / 0,587 | 74,2 % / 75,0 % | 60/66 · 18/38 | 13 (0) / 11 (2) |
| exp. comité de 11 redes | 60,6 % [41,7–77,4] | 74,2 % | 74,0 % / 0,610 | 75,8 % / 77,8 % | 63/66 · 23/38 | 14 (3) / 13 (2) |
| exp. segmentación v2 | 38,5 % [30,1–50,0] | 39,4 % | 71,2 % / 0,589 | 42,4 % / 43,3 % | 60/66 · 24/38 | 12 (2) / 9 (1) |
| exp. normalización (estirar) | 37,5 % [25,0–52,4] | 54,5 % | 50,0 % / 0,271 | 74,2 % / 73,5 % | 49/66 · 34/38 | 13 (0) / 8 (1) |

(*) El modelo entregado se entrenó con fotos que hoy están en el informe: no es una prueba limpia. Novedades en etapa 1:
0 en todas salvo principal (4) y medianas de la ronda 1 (7); categorías sin etiqueta: 0 salvo canónico (1 en calidad),
sin tope (1 en morfología, 2 en calidad) y segmentación v2 (2 en calidad).

**Variabilidad** (30 particiones internas; media ± desvío [mín–máx]; informe):

| variante | cadena total | cadena manzanas | tipo exactitud | otras frutas derivadas |
|---|---|---|---|---|
| original reentrenado | 34,2 % ± 5,2 [21–44] | 33,8 % ± 9,6 | 48,1 % ± 8,2 | 18,3 ± 5,9 |
| compatible mínima | 49,7 % ± 3,7 | 61,8 % ± 9,6 | 65,3 % ± 4,4 | 23,1 ± 6,5 |
| **principal** | **63,0 % ± 4,3 [55–73]** | 72,5 % ± 7,3 [56–88] | 76,0 % ± 3,9 | 18,6 ± 4,0 |
| alternativa canónica | 62,5 % ± 6,0 | 74,4 % ± 10,3 | 74,3 % ± 6,2 | 20,2 ± 4,5 |
| sin tope (exp.) | 68,4 % ± 4,0 | 75,0 % ± 5,8 | 79,3 % ± 2,9 | 15,1 ± 3,7 |

**Sensibilidad al orden** (ajuste final, 30 órdenes aleatorios, informe): principal 58,5 % ± 8,5 **[39–69]** (11–12
categorías); original reentrenado 31,6 % ± 3,3 [27–40]; comité 61,7 % ± 2,5. El orden canónico no varía (0,0 por
construcción: solo depende del conjunto de entrenamiento). **El 52,9 % de la principal es una realización del orden**:
la cifra a citar es la distribución, no ese punto. Mezclar con una semilla fija no elimina la sensibilidad.

**Matrices de confusión**: `comparacion.md`, sección D (tipo y calidad aislada para principal, original reentrenado y
medianas). **Diagnóstico basado solo en el fondo** (marco del 5 % del borde, 5 bits): tipo 59,6 % (30 particiones
61,4 % ± 3,1) y calidad 77,3 % (74,7 % ± 5,6) contra 67,3 % y 74,2 % de la principal.

**Categorías pequeñas**: la principal tiene 1 categoría de una sola muestra en morfología y 2 en calidad (tamaños
`comparacion.md`, sección A); el experimental sin tope tiene 12 de 45. Con ≤ 14 categorías la fragmentación es moderada;
la categoría de morfología etiquetada `citrico` (41 muestras) contiene 18 manzanas: **la separación cítrico/manzana con
estos bits es débil** (en el informe, 21 de 38 otras frutas se derivan a calidad).

## A7. Dependencia de fondo e iluminación — `resultados/ronda2/robustez_*.md`

`scripts/robustez.py` sustituye el fondo por colores lisos (máscara de GrabCut) y aplica gamma, ganancia y tinte a las
104 fotos del informe, y mide cuántas predicciones de la cadena **no cambian** respecto de la foto original.

| sistema | exactitud cadena (original) | estabilidad: fondos (media de 4) | estabilidad: iluminación (media de 6) |
|---|---|---|---|
| original entregado | 42,3 % | 46,9 % | 75,0 % |
| **principal** | 52,9 % | **50,2 %** | 64,9 % |
| alternativa orden canónico | 60,6 % | 52,6 % | 64,6 % |
| exp. segmentación v2 | 38,5 % | 67,1 % | 61,1 % |
| exp. normalización | 37,5 % | 41,3 % | 71,2 % |

* Con **fondo blanco o gris, el ~38 % de las predicciones de la principal cambian; con fondo azul, ~71 %.** Con
  iluminación distinta cambian ~35 %. Es coherente con el diagnóstico de fondo: el sistema **depende del escenario y de
  la exposición**, y esto no se corrigió.
* La segmentación v2 es **más estable ante el fondo** (67 %) pero menos exacta: con v1, parte de la exactitud sobre este
  banco viene de artefactos de segmentación propios de cada escenario. La normalización no mejoró la estabilidad frente
  al fondo y empeoró la exactitud.
* Es un diagnóstico (máscaras aproximadas, imágenes sintéticas): no prueba causalidad ni cubre capturas reales nuevas.

## A8. Recomendación

1. **Configuración principal compatible con el primer avance: `modelos_v2/principal/`** (elegida por la regla declarada,
   sin mirar el informe). Informar la **distribución** (63,0 % ± 4,3 en 30 particiones; 58,5 % ± 8,5 según el orden), no un
   punto.
2. **Alternativa para decidir con la cátedra: `alternativa_orden_canonico/`**: mismo desempeño (62,5 % ± 6,0 vs 63,0 %,
   diferencia no concluyente) y resultado determinista. Si el equipo valora la reproducibilidad por encima de los ~2 puntos de
   validación, es la mejor opción. No se eligió por el informe.
3. **Experimentos separados** (no son la configuración principal): sin tope de categorías (mejor desempeño medido,
   pero viola el criterio de éxito del primer avance), medianas, protuberancia, segmentación v2, normalización, comité.
4. **No afirmar** que el sistema "reconoce calidad de manzanas": con este banco no se puede distinguir de reconocer el
   escenario. Es una limitación para el informe final, no un detalle.

## A9. Reproducibilidad e instalación (punto 5)

* Versiones fijadas en `requirements*.txt`; `python scripts/verificar_instalacion.py` crea un entorno virtual temporal,
  instala las versiones fijas y ejecuta pruebas, demo y ejemplos; el registro queda en `resultados/verificacion/`.
* Determinismo comprobado: las mediciones de la segmentación v2 salen idénticas con distinta cantidad de procesos; el
  barrido y las comparaciones se repiten idénticos. `tests/test_modelos_v2.py` reproduce cada modelo desde su CSV.
* **Los originales no se regeneran ni sobrescriben**: `datasets/`, `modelos/`, `baseline_original/` intactos (`git status`
  no los lista); `scripts/final_ronda2.py` solo escribe en `datasets_v2/`, `modelos_v2/` y `config/`, rechaza pisar archivos
  existentes sin `--sobrescribir` y respalda la primera ronda con copia verificada por hash.
* La extracción, el entrenamiento, el guardado y la inferencia usan la misma configuración: la de
  `preprocesamiento_<etapa>.csv` / el modelo.

## A10. Pendientes

**De código (no dependen de nadie más)**
* Reducir la sensibilidad al orden dentro de ART1 (más allá de ordenar por actividad): p. ej. elegir el orden por
  validación anidada, o presentar los patrones por frecuencia.
* Un detector de fruta/objeto ajeno antes de ART1 (hoy un auto se clasifica como manzana).
* Etiquetado manual de categorías creadas por `--aprender` y actualización de `n`/pureza tras aprender.
* Evaluar descriptores cromáticos con otra definición de color uniforme por región (p. ej. por cuadrantes).

**Dependen del equipo o de datos nuevos**
* **Material de Lau** (`Lau.pp5.a.11`, `Lau.pp12.a.14`): no está en el repositorio; la equivalencia con su notación no se
  verificó (solo Carpenter-Grossberg y Fausett).
* **Procedencia y licencia del dataset** (origen Kaggle según la sesión histórica): no documentadas; hay que citarlas.
* **Nuevas capturas con identificador de fruta y de sesión**, la misma fruta en escenarios e iluminaciones distintos, y
  fotos "difíciles" que el curador del primer avance eliminó: es lo único que permite separar fruta de escenario y
  declarar independencia entre particiones.
* **Anotaciones de tallos** (aunque sea una muestra) para validar `tiene_tallo` y el detector por protuberancia.
* **Decisión de cátedra** sobre el criterio de < 15 categorías (cuesta 5–6 puntos) y sobre el orden determinista.
* **Portada, manual de referencia (PDF), informe de corridas (PDF) y PPT con audio**: son entregables de redacción del
  equipo; este informe y `resultados/` son su insumo.
* Decidir si `baseline_original/` y `SESION_DE_TRABAJO.md` (señalado como **histórico**) se conservan en la entrega.

## A11. Archivos de la segunda ronda

*Nuevos*: `scripts/{busqueda,comparar_ronda2,final_ronda2,robustez,verificar_instalacion}.py`,
`datasets_v2/<variante>/`, `modelos_v2/<variante>/`, `resultados/ronda2/`, `resultados/verificacion/`.
*Modificados*: `CarGross.py` (tipo de modelo, `--preprocesamiento`, `soporte_min`), `validacion.py`, `extraccion.py`
(opciones de color/normalización, CSV de preprocesamiento), `preprocesador.py` (descriptores de color, normalización),
`evaluacion/*` (informe reservado, orden, comité, derivación mínima), `probar_imagen.py`, `inferencia.py`, `demo.py`,
`requirements*.txt`, `README.md`, `SESION_DE_TRABAJO.md` (solo una nota de "histórico" al inicio).

---

# PARTE B — Primera ronda (histórico: protocolo exploratorio)

> Lo que sigue es el informe de la primera ronda. Sus cifras (p. ej. 60,2 % de la receta de medianas) provienen de un
> protocolo en el que **todas** las fotos intervinieron en validación y prueba y la variante de medianas era la
> configuración final. Se conserva por trazabilidad; las rutas `datasets_v2/morfologia_*.csv` y `modelos_v2/art1_*.json`
> que menciona están ahora en `datasets_v2/ronda1_medianas/` y `modelos_v2/ronda1_medianas/`.

## Informe de la primera ronda

Material para incorporar al manual y al informe final. Escrito para el equipo y la cátedra: cada afirmación
indica de dónde sale (archivo o comando). Las cifras son las de la última corrida de
`python scripts/experimentos.py todo` (Python 3.14, OpenCV 5.0.0.93, NumPy 2.4.6).

**El primer avance no se tocó**: `datasets/`, `modelos/`, las fotos y el código entregado se conservan
(`baseline_original/` es un snapshot del commit `f01b4dd`). Lo nuevo vive en carpetas distintas
(`datasets_v2/`, `modelos_v2/`, `resultados/`, `config/`). Se conservan la red de Carpenter-Grossberg
(`CarGross.py`), la arquitectura de dos etapas, la entrada CSV, las variables comprometidas, el preprocesamiento
(segmentación, centrado, escalado, extracción, binarización), el descriptor cromático sugerido por la docente,
las referencias `premium/comercial/descarte` y el alias `carozo` (= manzana).

---

## 1. Resumen

* La línea de base se **reprodujo exactamente** (14/76,11 %, 19/75,56 %, 17/81,11 %, 21/81,67 % y 167/339,
  96/219, 65/219, 44/120): `resultados/baseline_original.md`.
* Se encontró la **causa raíz de `tiene_tallo = 0`**: con OpenCV 5 `convexityDefects` devuelve un arreglo 2-D y
  el código del primer avance exigía 3-D; el bloque se salta en silencio y la variable vale 0 para todas las
  fotos. No es solo un umbral mal elegido.
* Se implementaron las seis mejoras pedidas (validación, configuración/persistencia, etiquetado con prototipos
  finales, cobertura de manzanas, aprendizaje explícito, errores/curación segura) con 200 pruebas.
* **Resultado, con cautelas**: en 30 particiones aleatorias por grupo, la cadena completa acierta 60,2 % ±
  7,0 en prueba contra 45,2 % ± 12,3 de la receta original reentrenada con los mismos datos; en las manzanas
  (tipo y calidad correctos) 63,5 % ± 10,1 contra 40,9 % ± 13,9 y contra 44,3 % ± 2,8 de la referencia
  "siempre manzana + calidad mayoritaria". La mejora viene sobre todo de
  **cobertura de manzanas de todas las calidades, etiquetado coherente, umbrales/ρ elegidos sin tocar la
  prueba, descriptores cromáticos adicionales y entrenar hasta estabilizar**. La nueva segmentación, el cambio
  de medición de `tiene_tallo` y el tono circular **no mostraron mejora medible** (§5).
* **Lo que no se resolvió y limita cualquier conclusión**: (a) el banco son 339 fotos ya filtradas por el
  curador del primer avance; (b) no hay identificador de fruta física ni de sesión; (c) las etiquetas de calidad
  están confundidas con el escenario de la foto: mirando **solo el borde de la imagen** se predice la calidad
  de la manzana con 77,6 % de exactitud, igual que el sistema (77,0 %) — `resultados/diagnostico_fondo.md`;
  (d) la separación cítrico/manzana sigue siendo floja.

---

## 2. Hallazgos recibidos: qué se comprobó

| hallazgo | estado | evidencia |
|---|---|---|
| Datasets de 180 registros; 6 y 5 variables | Confirmado | `baseline_original.md` §A |
| Corridas de referencia (14/76,11 %; 19/75,56 %; 17/81,11 %; 21/81,67 % ×2) | Confirmado, coinciden exactamente | §B |
| 167/339; 96/219; 65/219; 44/120 con los modelos entregados | Confirmado, coinciden exactamente | §C (son reinferencia: incluyen fotos de entrenamiento) |
| La etapa 1 se entrena solo con manzanas buenas | Confirmado | `carozo` sale solo de `Apple_Good` (trazabilidad `datasets_v2/trazabilidad_ids_originales.csv`, 360/360 filas verificadas). Con el modelo entregado se derivan a calidad 61/99 manzanas *good*, 24/60 *bad* y 11/60 *mixed* |
| `tiene_tallo` = 0 en todos los registros | Confirmado, **con causa raíz** (OpenCV 5, ver §4.4) | `tests/test_preprocesamiento.py::test_v1_queda_en_cero_cuando_opencv_devuelve_2d` |
| Vectores idénticos con etiquetas diferentes | Confirmado | morfología: 9 de 20 vectores distintos mezclan etiquetas (100 de 180 filas), techo de pureza 84,44 %; calidad: 10 de 23 (107 filas), techo 86,67 %. Ningún ρ supera ese techo |
| Mapeo con asignaciones online | Confirmado | con los prototipos finales, 75/180 muestras de morfología y 44/180 de calidad cambian de categoría; 5/19 y 2/17 categorías cambiarían de etiqueta o quedarían vacías. En estos datasets **ninguna** muestra queda sin resonancia al reevaluar |
| ρ = 0,85 en el modelo, 0,75 en rutas de regeneración | Confirmado | `probar_imagen.py:48` y `scripts/curar_dataset.py:342` usaban 0,75 |
| Umbrales de calidad redondeados | Confirmado | 3 de 900 bits difieren con los umbrales de lote exactos; además esas constantes se calcularon con las 180 manzanas, que incluyen fotos que hoy son de prueba |
| Columna con valor inválido pasa a metadatos | Confirmado | un `2` en una variable la convertía en metadato y la red se entrenaba con menos variables |
| Corrida interrumpida por límite de categorías sale con código 0 | Confirmado | `--max-categorias 5` terminaba con código 0 |
| `--aprender` incorpora novedades solo en memoria | Confirmado (lectura de código) | el modelo original nunca se guardaba |
| El curador elimina imágenes y sobrescribe archivos | Confirmado (lectura de código, **no se ejecutó**) | `os.remove` sobre las fotos y escritura sobre `datasets/` y `modelos/` |
| Ejemplos del README no incluidos | Parcial | los 5 `ejemplos/*.jpg` existen en este árbol; sí estaban rotas otras referencias (`datasets/dataset_calidad.csv` en el docstring de `CarGross.py`, scripts inexistentes en `SESION_DE_TRABAJO.md`, "150 frutas / 90 manzanas" en el README). `tests/test_readme_ejemplos.py` ahora ejecuta los ejemplos |

`SESION_DE_TRABAJO.md` (08/09) se dejó como registro histórico: describe scripts que ya no existen.

---

## 3. Cambios por área

### 3.1 Validación explícita de entradas (`validacion.py`)

* **Esquemas** `morfologia` y `calidad` con sus columnas, el identificador, las etiquetas válidas y los
  descriptores adicionales opcionales. Los CSV entregados validan sin cambios.
* **Errores con fila y columna**, todos juntos: archivo inexistente o vacío, sin filas, encabezado incorrecto o
  repetido, columnas desconocidas, valores ausentes, valores no binarios (`2`, `0.5`, `si`, `01`),
  identificadores vacíos o duplicados, etiquetas fuera de rango, filas con campos de más o de menos.
* Una característica inválida **ya no pasa a metadato**. Un vector nulo genera una advertencia (el dataset de
  calidad tiene 16).
* **Modo genérico explícito** (`--esquema generico --col-etiqueta X [--col-meta Y]`): toda columna no declarada
  debe ser binaria; las columnas `*_real` no declaradas se rechazan, así una etiqueta no entra como variable.
* **Compatibilidad con el modelo**: `validar_compatibilidad` / `reordenar_a_modelo` comprueban cantidad y nombres
  de variables y alinean por nombre, no por posición.

### 3.2 Configuración coherente y persistencia

* `config/parametros_etapas.csv` es la única fuente de ρ, L, tope, pasadas, política de vector nulo y
  desempate para entrenamiento, regeneración y demo. La inferencia usa **lo guardado en el modelo**.
* **Modelo de formato 2** (`CarGross.RedART1.guardar`): versión, ρ, L, política de nulo, orden de variables y
  `preprocesamiento` (segmentador, tamaño de canvas, reglas y **umbrales exactos**), mapeo de categorías, hash del
  CSV de entrenamiento y procedencia. Escritura atómica; no pisa archivos sin `--sobrescribir`. Se **valida al
  cargar** (15 condiciones, cada una con prueba).
* **Compatibilidad**: los modelos entregados (formato 1) se cargan en *modo legado*: mismas etiquetas, política
  de nulo del primer avance y preprocesamiento v1, con una advertencia explícita; el archivo no se modifica.
  `CarGross.py --migrar-modelo` genera un formato 2 **con los mismos pesos** y las etiquetas recalculadas con
  esos prototipos. `probar_imagen.py --legado` usa los modelos entregados.
* Los modelos ya no se entrenan en silencio: antes, `probar_imagen.py` entrenaba y guardaba si faltaba un modelo.
  Ahora falla con código 3 e indica cómo generarlo.

### 3.3 Etiquetado de categorías con prototipos finales (`CarGross.construir_mapeo`)

Cada muestra se vuelve a presentar en modo inferencia con los prototipos finales (la misma competencia y
vigilancia del programa); la etiqueta es la mayoritaria entre las que resuenan. Tratamiento explícito de:
muestras sin resonancia (no votan, se cuentan), categorías sin muestras (sin etiqueta), empates
(`indeterminada`: sin etiqueta; `mayoritaria_global`: por frecuencia de todo el entrenamiento) y vector nulo
(categoría propia o rechazo, según política). Se conserva la separación agrupamiento / etiquetado. Detalle y
tabla de casos: `docs/que_aprende_art1.md` §3. El mapeo del primer avance sigue disponible (`--mapeo online`)
solo para comparar.

*Decisión de empates*: `mayoritaria_global` mejoró la validación (+0,021 ± 0,006) y la prueba (+0,024) frente a
dejar la categoría sin etiqueta, por lo que es el valor de `config/`. Es un desempate entre etiquetas
observadas, no una etiqueta inventada; las categorías creadas por aprendizaje en línea siguen sin etiqueta.

### 3.4 Representación y cobertura de las manzanas

* **Cobertura**: la etapa 1 se entrena con manzanas de las tres calidades (`carozo`). Es la mejora medida de
  mayor efecto aislado entre las de método (+0,051 en validación, +0,048 en prueba). Con ella entran también
  las 39 manzanas *good* que el curador había descartado como `carozo` por no ser esféricas/lisas.
* **`tiene_tallo`**: ver §4.4. La medición nueva (protuberancia fina del contorno) deja de ser una constante,
  pero **no está validada contra una verdad de terreno de tallos**: no existe, y a ojo las máscaras suelen
  borrar el tallo o incluir fondo. Se conserva como variable comprometida, con esa salvedad.
* **Tono circular**: `desvio_hue` del primer avance era la desviación lineal del tono de OpenCV (0–179), que
  es cíclico: los rojos están en ambos extremos y salían "heterogéneos". `desvio_tono` usa estadística
  circular ponderada por saturación. Corrige un defecto de medición demostrado en `tests/`; en las métricas fue
  neutral (+/- dentro del ruido).
* **Descriptores cromáticos adicionales** (sugerencia docente), cada uno medido por separado (`ablacion.csv`):

  | descriptor | etapa | aporte solo (Δ cadena en validación ± EE) | en prueba |
  |---|---|---|---|
  | `tono_calido` (distancia del tono medio al naranja ≤ umbral) | 1 | +0,052 ± 0,011 | +0,042 |
  | `saturacion_alta` (saturación media ≥ umbral) | 1 | +0,028 ± 0,010 | +0,022 |
  | ambos | 1 | +0,053 ± 0,013 | +0,041 |
  | `sin_oscurecimiento` (fracción de píxeles muy oscuros ≤ umbral) | 2 | +0,022 ± 0,006 | +0,022 |

  Se incorporaron los tres. Advertencia: el color de las frutas del banco (naranjas anaranjadas, bananas
  amarillas) puede ser un atajo que no generalice a otras variedades.
* **Segmentación v2** (candidatos Lab / GrabCut / v1 con puntaje de plausibilidad sin etiquetas): a simple
  vista sigue mejor el contorno en bananas, pasto y manos con guantes
  (`docs/figuras/segmentacion_v1_v2.png`), pero **no mejora la métrica** (§5). No se adoptó como valor por
  defecto; queda disponible (`segmentador: v2` en la configuración).
* **Criterios de exclusión**: **no se excluyó ninguna foto**. Las 339 que existen ya son las que sobrevivieron al
  curador (cuyos filtros, documentados en `scripts/curar_dataset.py`, usan medidas parecidas a las de la red: el
  banco es más fácil que el uso real). La única exclusión técnica posible (imagen ilegible o máscara vacía)
  no ocurrió en ninguna foto. Ninguna foto quedó marcada como segmentación dudosa por el puntaje de plausibilidad
  (mínimo 0,57), lo que muestra que ese puntaje **no** es un filtro de calidad fuerte: hay fotos "plausibles"
  con varias manzanas juntas (las *mixed*).

### 3.5 Aprendizaje e inferencia diferenciados (`inferencia.py`, `probar_imagen.py`)

* La inferencia llama solo a `RedART1.evaluar`; se verifica que la huella de los pesos no cambia (también en el
  archivo del modelo).
* `--aprender`: plasticidad **en memoria**; resonancia → prototipo AND; sin resonancia → categoría nueva
  **sin etiqueta** (`nueva_sin_etiqueta`); cada evento queda en `historial_aprendizaje`. Se guarda solo con
  `--guardar-aprendizaje CARPETA`, en archivos nuevos; nunca sobre el modelo de origen ni sobre un archivo
  existente. Los modelos de formato 1 deben migrarse antes de guardar.
* La salida llama **coincidencia ART1** a `|I∧T|/|I|` y aclara que no es probabilidad de acierto; la **pureza
  de entrenamiento** se presenta como mezcla de la categoría, no como confianza de la foto. Se agrega la
  advertencia de que ART1 no rechaza objetos ajenos al dominio (`ejemplos/auto.jpg` se clasifica como manzana
  y se deriva a calidad: es el ejemplo).

### 3.6 Errores, ayuda y curación segura

* Códigos de salida: `0` ok · `1` corrida infructuosa · `2` argumentos · `3` entrada inválida o ausente · `4`
  error interno. Cada fallo imprime un diagnóstico y cómo pedir ayuda; `--ayuda`/`--help` en todos los programas.
* `demo.py` propaga los fallos (código 1) y ya no genera ni borra datasets.
* **`scripts/curar_dataset.py`**: `seleccionar` (evalúa todas las fotos con los criterios del primer avance,
  escribe `salida_curacion/` y no toca nada) separado de `eliminar` (simulación por defecto; con `--ejecutar`
  **mueve** a una carpeta; borrar exige además `--borrar-definitivamente --confirmo-que-tengo-copia`). No pisa
  salidas existentes. Sobre las fotos actuales reproduce exactamente `datasets/*.csv` (misma selección y orden).
  **No se ejecutó ninguna eliminación.**

---

## 4. Detalles técnicos que conviene defender

### 4.1 Núcleo neuronal sin cambios de comportamiento
`RedART1.clasificar` ahora delega la competencia en `evaluar` (una sola implementación del ciclo) y la entrada
neta se calcula como una única división, para que el desempate entre categorías sea determinista. Se comprobó
que **no cambia nada**: 132 corridas (2 datasets × 11 valores de ρ × 6 órdenes) dan asignaciones **y**
prototipos idénticos a los del código entregado (`tests/test_preprocesamiento.py::test_nucleo_art1_equivale_al_entregado`).

### 4.2 Vector nulo
Para `|I| = 0` el match `|I∧T|/|I|` no está definido. Política `categoria_propia` (la del primer avance,
conservada por compatibilidad y por los datos: 16 de 180 manzanas tienen todas las variables en 0): resuena solo
con un prototipo nulo (match 1,0 por convención) y no absorbe patrones no nulos. Política `rechazar`: no se
aprende (error) ni resuena (novedad). Es una convención del proyecto, no un resultado de la bibliografía.

### 4.3 Match asimétrico
Con `I = 110000` y `T = 111100`, `|I∧T|/|I| = 1`; con los papeles invertidos, `0,5`. Un match de 100 % indica
que el patrón está **contenido** en el prototipo, no que sean iguales ni que la etiqueta sea correcta.

### 4.4 Causa raíz de `tiene_tallo = 0`
`extraer_features` (v1) calcula la profundidad del mayor defecto de convexidad solo si `defectos.ndim == 3`. Con
OpenCV 4.x la forma es `(N,1,4)`; con **OpenCV 5.x** (la instalada) es `(N,4)`: la condición falla, el bloque se
omite sin error y `profundidad_defecto = 0.0` en las 339 fotos. Además, aun corregida, esa medida mide
*hendiduras*, no un apéndice. Se agregaron `profundidad_defecto_v2` (independiente de la versión) y `medir_tallo`
(protuberancia fina). Consecuencia para la reproducibilidad: **con OpenCV 4 los datasets/modelos entregados no se
regenerarían idénticos**; por eso `requirements.txt` registra las versiones usadas.

### 4.5 Particiones y fuga
No hay identificador de fruta física ni de sesión. Como proxy, `evaluacion/banco.py` agrupa fotos de la misma
carpeta tomadas con ≤ 3 s de diferencia (según el nombre de archivo) y reparte **grupos completos** entre
entrenamiento/validación/prueba, estratificado por origen. Reduce la fuga, **no demuestra independencia**: dos
frutas distintas fotografiadas en la misma sesión (misma luz y fondo) pueden quedar en particiones distintas.
Los umbrales dependientes de datos se calculan **solo con el entrenamiento**; ρ se elige en validación; la
prueba no interviene en ninguna decisión. Dos pruebas lo comprueban perturbando las fotos de prueba/validación
y verificando que umbrales, pesos y ρ no cambian (`tests/test_evaluacion.py`).

---

## 5. Evaluación

Parámetros y datos: `resultados/comparacion.md` (versión completa) y `resultados/ablacion.csv`.

### 5.1 Original vs. mejorada

*Original entregado* = `modelos/` + constantes del primer avance. *Original reentrenado* = la misma receta
(segmentación v1, constantes, asignación online, etapa 1 solo con manzanas *good*, ρ 0,85/0,70, una pasada)
entrenada solo con la partición de entrenamiento. *Mejorada* = configuración elegida (abajo).

**A. Diagnóstico sobre las 339 fotos (reinferencia, incluye fotos de entrenamiento; no es validación externa)**

| | original entregado | mejorada |
|---|---|---|
| tipo correcto | 167/339 (49,3 %) | 255/339 (75,2 %) |
| manzanas derivadas a calidad | 96/219 | 170/219 |
| tipo y calidad correctos (manzanas) | 65/219 (29,7 %) | 128/219 (58,4 %) |
| naranjas/bananas derivadas a calidad | 44/120 | 35/120 |
| referencia "siempre carozo" (tipo) | 64,6 % | 64,6 % |

El original entregado no supera "siempre manzana" en tipo (49,3 % < 64,6 %).

**B. Prueba en la partición principal (semilla 0, 78 fotos; entrenamiento 190, validación 71)**

| | original entregado (*) | original reentrenado | mejorada |
|---|---|---|---|
| tipo correcto | 64,1 % | 55,1 % | 60,3 % |
| F1 macro de tipo | 0,618 | 0,548 | 0,546 |
| manzanas derivadas | 29/51 | 29/51 | 34/51 |
| naranjas/bananas derivadas | 6/27 | 13/27 | 14/27 |
| calidad aislada (exactitud) | 88,2 % | 72,5 % | 90,2 % |
| tipo y calidad correctos (manzanas) | 23/51 | 21/51 | 30/51 |
| cadena total | 56,4 % | 44,9 % | 55,1 % (IC95 42,9–68,2) |
| referencia mayoritaria (tipo) | 65,4 % | | |

(*) Entrenado con fotos que hoy están en esta partición: **no es una prueba limpia**, es optimista.

En esta partición la mejorada **no supera "siempre manzana" en exactitud de tipo** (60,3 % < 65,4 %), aunque sí
en F1 macro (0,546 > 0,264), y deriva a calidad a 14 de 27 naranjas/bananas. Es una sola partición de 78 fotos.

**C. 30 particiones aleatorias por grupo (media ± desvío; las particiones comparten fotos: no es un IC)**

| prueba | original reentrenado | mejorada | referencia mayoritaria |
|---|---|---|---|
| cadena total (todas las fotos) | 45,2 % ± 12,3 | 60,2 % ± 7,0 | – |
| cadena de manzanas (tipo + calidad) | 40,9 % ± 13,9 | 63,5 % ± 10,1 | 44,3 % ± 2,8 |
| tipo: exactitud | 57,7 % ± 14,6 | 72,3 % ± 8,9 | 63,8 % ± 6,0 |
| tipo: F1 macro | 0,514 ± 0,143 | 0,653 ± 0,110 | ≈ 0,26 |
| calidad aislada: F1 macro | 0,638 ± 0,097 | 0,748 ± 0,131 | – |

La variabilidad entre particiones (7–12 puntos) es del tamaño de varias mejoras individuales: por eso la
decisión entre variantes se tomó promediando 30 particiones y con diferencias pareadas.

### 5.2 Cadena de decisiones (validación; `resultados/ablacion.csv`)

Regla declarada de antemano: *obligatoria* (corrección de método pedida) se adopta siempre; *corrección* de un
defecto demostrado se adopta si no es peor que −0,01; *mejora* solo si ≥ +0,01 y mayor que 1 error estándar de
la diferencia pareada. Métrica: exactitud de la cadena completa. EE = desvío de las diferencias entre
particiones (descriptivo, no un IC independiente).

| paso | decisión | Δ validación ± EE | Δ prueba |
|---|---|---|---|
| S1 etiquetado con prototipos finales | adoptada (obligatoria) | +0,030 ± 0,020 | −0,016 |
| S2 desempate por frecuencia global | adoptada (mejora) | +0,021 ± 0,006 | +0,024 |
| S3 manzanas de todas las calidades en etapa 1 | adoptada (obligatoria) | +0,051 ± 0,018 | +0,048 |
| S4 ρ elegido en validación | adoptada (obligatoria) | +0,040 ± 0,008 | 0,000 |
| S5 umbrales de calidad = medianas exactas del entrenamiento | adoptada (obligatoria) | −0,002 ± 0,006 | −0,009 |
| S6 segmentación v2 | **no adoptada** (sin evidencia) | −0,001 ± 0,016 | −0,042 |
| S7 `tiene_tallo`: protuberancia (vs. convexidad corregida −0,010) | adoptada (corrección) | +0,001 ± 0,007 | −0,007 |
| S8 tono circular | adoptada (corrección) | −0,004 ± 0,012 | +0,011 |
| S9 umbrales de morfología = mediana del entrenamiento | adoptada (mejora) | +0,026 ± 0,009 | **−0,012** |
| S10 `tono_calido` + `saturacion_alta` | adoptada (mejora) | +0,053 ± 0,013 | +0,041 |
| S11 `sin_oscurecimiento` (calidad) | adoptada (mejora) | +0,022 ± 0,006 | +0,022 |
| S12 hasta estabilizar (máx. 10 pasadas) | adoptada (mejora) | +0,051 ± 0,012 | +0,047 |
| S13 segmentación v2 (reconsiderada al final) | **no adoptada** | −0,030 ± 0,016 | −0,011 |

Lecturas honestas:

* **S1, S5, S7 y S8 son neutrales en métrica**; se adoptaron porque corrigen un defecto o un principio de
  método (coherencia con la inferencia, umbrales solo con entrenamiento, variable que valía siempre 0, tono
  cíclico), no porque mejoren la cifra.
* **S9** (umbrales por mediana) mejoró en validación pero **no en prueba** (−0,012): evidencia insuficiente de
  que generalice. Tiene además un costo de interpretación: `es_esferica = 1` pasa a significar "más redonda que
  la mediana del entrenamiento". Si el equipo prefiere las constantes absolutas del primer avance, el costo
  medido es de unos 2 puntos en validación (ver `ablacion.csv`, S9).
* **La segmentación v2 no ayudó** a pesar de verse mejor. Hipótesis (no demostrada): con v1, las máscaras
  fragmentadas de las fotos *mixed* (pasto, varias manzanas) producen un patrón de bits propio de ese escenario
  que ayuda a separarlas; v2 lo "corrige". Es otra señal de la confusión escenario-clase (§6).
* El efecto individual de cada descriptor cromático está en `ablacion.csv` (`S10`, `S11`).

**Configuración elegida** (`resultados/config_final.json`; modelos en `modelos_v2/`):
segmentación v1; `tiene_tallo` por protuberancia; tono circular; umbrales de morfología y calidad por mediana
del entrenamiento; +`tono_calido`, `saturacion_alta`, `sin_oscurecimiento`; etapa 1 con manzanas de todas las
calidades; etiquetado con prototipos finales y desempate global; hasta 10 pasadas; ρ por validación
(partición principal: 0,55 en morfología, 0,45 en calidad; promedio de las 30 particiones: 0,65 y 0,53).

### 5.3 Barrido de ρ y sensibilidad al orden (`resultados/barrido_orden.md`)

* En los datasets entregados, con 31 órdenes, el número de categorías varía poco entre órdenes (p. ej. ρ = 0,85
  en morfología: 17–20) y la pureza online 73–82 %; ρ ≥ 0,85 da el **mismo** resultado (los valores efectivos de
  ρ son discretos porque el match toma pocos valores con 5–6 bits). Ningún ρ supera el techo de pureza por
  vectores idénticos (84,44 % / 86,67 %).
* En la versión mejorada la etapa 1 (8 variables, 69 vectores distintos) **prolifera** para ρ ≥ 0,80: las 31
  corridas superan el tope de 50 categorías (corrida infructuosa, código 1).
* Con la partición y ρ fijos, solo cambiar el orden de presentación mueve la cadena de prueba entre
  38,5 % y 69,2 % (media 50,8 % ± 7,1) en la versión mejorada y entre 32,1 % y 52,6 % en la original
  reentrenada. **Mezclar con una semilla fija no elimina la sensibilidad al orden**; la cifra de la partición
  principal (55,1 %) es una de esas posibilidades.
* Número de categorías: la versión mejorada usa **26** categorías de morfología (8 con una sola muestra) y 11 de
  calidad, para 3 clases: ART1 no se forzó a 3 grupos y, con estos bits, se comporta cerca de una tabla de
  consulta. Más categorías mejoran la pureza de entrenamiento pero no garantizan mejor prueba.

---

## 6. Limitaciones y cuidados con las conclusiones

1. **Banco filtrado.** Las 339 fotos son lo que el curador del primer avance dejó (criterios con medidas
   parecidas a las de la red). El desempeño real sería peor. No hay fotos "difíciles" borradas que se puedan
   recuperar (el curador las eliminó).
2. **Sin identificador de fruta física ni de sesión.** La agrupación por ráfagas es un proxy; no se afirma
   independencia entre particiones.
3. **Confusión escenario-clase** (`resultados/diagnostico_fondo.md`). El 97 % de los píxeles del marco de las
   fotos *mixed* es pasto; ninguno de las *bad* lo es (guantes, fondos claros). Con **5 bits del borde de la
   foto** se predice la calidad de la manzana con 77,6 % de exactitud (F1 macro 0,79) contra 77,0 % (0,75) del
   sistema, y el tipo con 67,2 % contra 72,3 %. Parte de lo que el sistema "acierta" puede ser reconocer el
   escenario. No hay forma de separar ambos efectos sin fotos de la misma fruta en escenarios distintos.
4. **Cítricos vs. manzanas.** En la prueba principal, 8 de 13 naranjas se clasifican como manzana y a 14 de 27
   naranjas/bananas se les aplica control de calidad. La cadena sigue derivando fruta que no es manzana.
5. **`tiene_tallo`** no está validada contra tallos reales.
6. **Un match de 100 % no garantiza etiqueta correcta ni rechazo de objetos ajenos** (auto → "manzana premium").
7. **Cambiar ρ no puede separar vectores idénticos**; las mejoras de ρ no superan el techo de pureza de los
   datos.
8. **Una prueba sintética exitosa no demuestra que los errores reales vengan del ruido**: las pruebas de
   `tests/` verifican el comportamiento del código, no la calidad de las predicciones sobre fotos reales.
9. **Las desviaciones y errores estándar de este informe describen la variación entre particiones que
   comparten fotos; no son intervalos de confianza independientes.** El IC95 por grupos de la partición
   principal es ancho (42,9–68,2 %).
10. **Reproducibilidad con otra versión de OpenCV**: con 4.x, `tiene_tallo` del primer avance se comportaría
    distinto (§4.4) y las mediciones podrían diferir levemente.
11. **Verificación bibliográfica incompleta**: falta el material de Lau; se contrastó con Carpenter-Grossberg
    (1987) y Fausett. No se declara equivalencia con Lau.

---

## 7. Pendientes sugeridos (para manual e informe final)

* **Obtener o fotografiar datos con identificador de fruta/sesión** y con la misma fruta en escenarios
  distintos; es lo único que permite separar fruta de escenario y declarar independencia.
* **Incorporar el material de Lau** y revisar la notación (competencia, vigilancia, aprendizaje, nodos no
  comprometidos).
* **Etiquetar los tallos a mano** en una muestra para validar `tiene_tallo`; evaluar un detector de tallo sobre
  la imagen y no sobre la máscara.
* **Mejorar la separación cítrico/manzana** con descriptores permitidos por la consigna (forma + color); medir
  cada uno con `scripts/experimentos.py`.
* **Decidir con la cátedra** si los umbrales relativos (mediana de entrenamiento) son aceptables para las
  variables de morfología o si se prefieren los absolutos del primer avance (costo medido en §5.2).
* **Etiquetado manual de categorías nuevas** creadas por `--aprender` (hoy quedan sin etiqueta) y un
  procedimiento para actualizar `n`/pureza tras aprender.
* **Rechazo de objetos ajenos** (p. ej. umbral de segmentación + control de tamaño/forma) como etapa previa;
  ART1 no lo resuelve.
* **Limpieza del repositorio**: decidir si `baseline_original/` y `SESION_DE_TRABAJO.md` se conservan en la
  entrega final.

---

## 8. Archivos nuevos y modificados

*Modificados*: `CarGross.py` (núcleo intacto en comportamiento; validación, política de nulo, mapeo final,
formato 2, CLI), `preprocesador.py` (v1 intacto; se agregó v2 al final), `probar_imagen.py`, `demo.py`,
`scripts/curar_dataset.py`, `README.md`, `requirements.txt`, `.gitignore`.
*Sin cambios*: `preprocesador_calidad.py`, `datasets/`, `modelos/`, `ejemplos/`.
*Nuevos*: `validacion.py`, `configuracion.py`, `extraccion.py`, `inferencia.py`, `evaluacion/`,
`scripts/{baseline_original,extraer_mediciones,experimentos,trazar_ids_originales,figura_segmentacion}.py`,
`tests/`, `config/`, `datasets_v2/`, `modelos_v2/`, `resultados/`, `baseline_original/`, `docs/`.
