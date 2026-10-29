# config/

## parametros_etapas.csv

Fuente unica de los parametros de **entrenamiento** de cada etapa; la leen `CarGross.py` (cuando se omiten `--rho`, `--L`, ...
con un dataset de esquema conocido), `demo.py` y `scripts/final_ronda2.py` (que lo reescribe con los de la variante principal).
La **inferencia no lo lee**: usa lo guardado en el modelo (`rho`, `L`, politica de nulo, preprocesamiento).

| columna | significado |
|---|---|
| `etapa` | `morfologia` o `calidad` |
| `rho` | vigilancia, en (0, 1]; variante principal: mediana de los elegidos en validacion interna, con <= 14 categorias |
| `L` | parametro de aprendizaje rapido (> 1) |
| `max_categorias` | tope duro; al alcanzarlo la corrida es infructuosa (codigo 1) |
| `epocas` | maximo de pasadas por los datos |
| `politica_nulo` | `categoria_propia` o `rechazar` |
| `desempate` | `indeterminada` o `mayoritaria_global` |
| `fuente` | de donde salio el valor |

El **orden de presentacion** no es un parametro: esta fijado en el orden de filas del CSV de entrenamiento
(`datasets_v2/<variante>/*_entrenamiento.csv`), por eso `CarGross.py --orden original` reproduce el modelo.
