# Qué aprende ART1 y qué decide el preprocesamiento

Documento para el manual y el informe final. Todo lo que sigue se puede comprobar en el código
(`CarGross.py`, `extraccion.py`, `preprocesador.py`) y en las pruebas (`tests/`).

## 1. Qué aprende la red

ART1 (Carpenter y Grossberg, 1987) recibe **vectores binarios** y forma **categorías**. Cada categoría
`j` es un **prototipo binario** `t_j` (los pesos top-down). En esta implementación (aprendizaje rápido, `L = 2`):

1. **Competencia.** Para un patrón `I`, cada categoría recibe la entrada neta `L·|I ∧ t_j| / (L − 1 + |t_j|)`.
   Compite primero la de mayor entrada neta (empate exacto: gana la de menor índice).
2. **Vigilancia.** Se calcula `|I ∧ t_j| / |I|`. Si es `≥ ρ` hay **resonancia**; si no, **reset**: esa categoría
   queda descartada para este patrón y compite la siguiente.
3. **Aprendizaje.** Con resonancia, `t_j ← I ∧ t_j` (el prototipo conserva solo los bits que comparten todos
   los patrones que le pertenecieron). Si ninguna categoría resuena, se **compromete una categoría nueva**
   con `t = I`.

### Ejemplo

Con `ρ = 0.7` y tres patrones de 6 bits:

| patrón | efecto |
|---|---|
| `111111` | no hay categorías: se crea la categoría 0 con prototipo `111111` |
| `111100` | `|I∧t|/|I| = 4/4 = 1 ≥ 0.7`: resuena, el prototipo pasa a `111100` |
| `110000` | `2/2 = 1`: resuena, el prototipo pasa a `110000` |

El prototipo final `110000` ya **no** resuena con el primer patrón (`2/6 = 0.33 < 0.7`) ni con el segundo
(`2/4 = 0.5`). Esto es lo que pasaba entre el aprendizaje y la etiqueta de las categorías en el primer
avance y por qué ahora se etiqueta con los prototipos finales (sección 3).

### Qué NO aprende

* **No aprende qué es una manzana, un cítrico ni qué es calidad.** Solo agrupa patrones de bits parecidos.
* **No usa las etiquetas** (`tipo_real`, `calidad_real`): no intervienen en la competencia, la vigilancia ni
  el aprendizaje. Existe separación estricta entre agrupamiento no supervisado y etiquetado posterior
  (`CarGross.construir_mapeo`).
* **No puede separar vectores idénticos.** Dos fotos con los mismos bits caen siempre en la misma categoría,
  tengan la etiqueta que tengan; cambiar `ρ` no lo evita. En los datasets entregados, por ejemplo, 100 de
  180 filas de morfología comparten su vector con filas de otra clase (techo de pureza 84,44 %; ver
  `resultados/baseline_original.md`).
* **El match no es una probabilidad ni una igualdad.** `|I ∧ t|/|I|` es asimétrico (un patrón puede estar
  contenido al 100 % en un prototipo muy distinto de él) y no mide la probabilidad de acertar la etiqueta.
  Un match de 100 % tampoco rechaza objetos ajenos al dominio: ART1 no sabe si la imagen es una fruta
  (`python probar_imagen.py --imagen ejemplos/auto.jpg` es el ejemplo: un auto resuena con una categoría
  "carozo" y se deriva a calidad).
* **Depende del orden de presentación.** Es un algoritmo en línea: otro orden puede dar otras categorías
  (`resultados/barrido_orden.md` cuantifica la variación). Mezclar con una semilla fija solo hace repetible
  un orden.

## 2. Qué decide el preprocesamiento (y no la red)

ART1 solo ve bits. **Todo lo que convierte una foto en bits lo decide el preprocesamiento**, y estas decisiones
pesan tanto o más que `ρ`:

| decisión | dónde vive | quién la fija |
|---|---|---|
| qué región es la fruta | `segmentar` (v1) / `segmentar_v2` | código (heurística, sin etiquetas) |
| centrado y escala | `centrar_y_escalar` (canvas 200×200) | código |
| qué se mide (aspecto, textura, solidez, gris, tono, saturación…) | `extraccion.medir_foto` | código |
| a partir de qué valor un bit es 1 | `umbrales` y `reglas` guardados **en el modelo** | variante principal: cortes **originales** en morfología; medianas del entrenamiento en calidad y en los descriptores adicionales |
| `ρ`, `L`, número de pasadas, tope de categorías | `config/parametros_etapas.csv` y, desde allí, el modelo | `ρ` se elige en validación, con el tope de < 15 categorías del primer avance |
| orden de presentación de los patrones | orden de filas del CSV de entrenamiento | variante principal: menos bits activos primero (la red es en línea: el orden cambia el resultado) |
| el preprocesamiento exacto de cada modelo | `preprocesamiento_<etapa>.csv` y sección `preprocesamiento` del modelo | se recupera/valida al guardar y al cargar |
| qué etiqueta lleva una categoría | `mapeo_clusters` del modelo | mayoría (con los prototipos finales) de las etiquetas de entrenamiento |
| cuándo una foto pasa a la etapa 2 | `inferencia.inferir_imagen` | regla del sistema: la etapa 1 devolvió la etiqueta `carozo` |

Consecuencias que conviene tener presentes al defender el trabajo:

* Un cambio de umbral cambia los bits y por lo tanto **qué patrones son idénticos**; es una decisión del
  preprocesamiento, no un ajuste de la red.
* Con umbrales por mediana del entrenamiento, `es_esferica = 1` significa "más redonda que la mediana de las
  fotos de entrenamiento", no "esférica" en sentido absoluto. Es la misma lógica de binarización relativa
  que el primer avance ya usaba en calidad.
* `densidad_alta`, `es_pesada` y `textura_firme` son **aproximaciones visuales** (solidez del contorno,
  área × solidez, varianza del laplaciano). No miden densidad física, gramos ni firmeza mecánica.
* `carozo` es el alias histórico de **manzana**; se conserva en datos, modelos y código.

## 3. Etiquetado de categorías

Después de entrenar, cada muestra de entrenamiento se vuelve a presentar **en modo inferencia**
(`RedART1.evaluar`, que no modifica pesos) con los prototipos finales, es decir, con la misma competencia y
vigilancia que usará el programa. La etiqueta de una categoría es la mayoritaria entre las muestras que
resuenan con ella. Casos explícitos:

| caso | tratamiento |
|---|---|
| muestra sin resonancia con los prototipos finales | no vota por ninguna categoría; se informa (`n_sin_resonancia`) |
| categoría sin muestras | `etiqueta = null`, estado `sin_muestras` |
| empate de etiquetas | `desempate = indeterminada`: `etiqueta = null`, estado `empate`; `mayoritaria_global`: gana la etiqueta más frecuente de todo el entrenamiento (estado `etiquetada_por_desempate`) |
| categoría creada por aprendizaje en línea | `etiqueta = null`, estado `nueva_sin_etiqueta`: no se inventa un tipo ni una calidad |

Una categoría sin etiqueta no se traduce a fruta ni calidad y no deriva a la etapa 2.

## 4. Pureza, match y confianza: tres cosas distintas

* **Pureza** (de entrenamiento): fracción de la etiqueta mayoritaria dentro de una categoría. Describe la
  mezcla al entrenar. No es la confianza en una foto individual.
* **Coincidencia / match**: `|I ∧ t|/|I|` de una foto contra un prototipo. No es una probabilidad de acierto.
* **Exactitud, F1 y cadena**: se miden con fotos **no usadas para entrenar** (partición de prueba).

## 5. Verificación contra la bibliografía

El núcleo (competencia, vigilancia, reset, resonancia, aprendizaje rápido) es el del primer avance y se
contrastó con la formulación de Carpenter y Grossberg (1987) y con Fausett: `b_ij = L·t_ji/(L − 1 + |t_j|)`,
vigilancia `|I ∧ t|/|I| ≥ ρ`, `t ← I ∧ t`. La única diferencia de formulación es el tratamiento de los nodos no
comprometidos: aquí no compiten, lo cual equivale a elegir pesos bottom-up iniciales arbitrariamente chicos,
algo que esa bibliografía permite (`0 < b_ij(0) < L/(L − 1 + n)`).

**Limitación:** el material de Lau exigido por la cátedra (`Lau.pp5.a.11`, `Lau.pp12.a.14`) **no está en el
repositorio**, por lo que no se comprobó la equivalencia con su notación. Queda como pendiente.
`tests/test_preprocesamiento.py::test_nucleo_art1_equivale_al_entregado` comprueba que el núcleo actual
produce exactamente las mismas asignaciones y prototipos que el entregado (132 corridas con distintos `ρ` y
órdenes).
