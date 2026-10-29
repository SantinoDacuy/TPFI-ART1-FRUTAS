# datasets_v2/

Generado por `python scripts/final_ronda2.py` (y `scripts/trazar_ids_originales.py`). Los datasets entregados del primer avance
siguen intactos en `datasets/`. Todo es CSV UTF-8 con encabezado; las columnas de variables son estrictamente 0/1.

```
manifiesto_imagenes.csv              una fila por foto del banco (339) con su bloque (desarrollo / informe)
trazabilidad_ids_originales.csv      fruta_id de los CSV entregados -> foto (verificado fila por fila, 360/360)
principal/                           variante principal compatible con el primer avance
alternativa_orden_canonico/          igual, con orden de presentacion determinista
experimental_medianas/  experimental_segmentacion_v2/  experimental_normalizacion/
ronda1_medianas/                     RESPALDO de la primera ronda (CSV, umbrales y manifiesto originales)
```

## Dentro de cada variante

| archivo | contenido |
|---|---|
| `{morfologia,calidad}_entrenamiento.csv` | entrada de la red: **solo fotos de desarrollo**, en el **orden de presentacion** usado para entrenar |
| `{morfologia,calidad}_informe.csv` | fotos del bloque de informe (no se usan para entrenar ni elegir); sirve para `CarGross.py --inferir-con` |
| `preprocesamiento_{morfologia,calidad}.csv` | **metadato de entrada en CSV**: como se convirtio cada foto en esas columnas |

Columnas de los datasets: `fruta_id` (identificador de **foto**, `F001`..., no de fruta fisica), las variables comprometidas,
los descriptores adicionales (`croma_alto`, `naranja_dominante` en morfologia; `croma_alto` en calidad) y la etiqueta de
referencia `tipo_real` / `calidad_real` (**nunca entra a la red**). La etapa de calidad contiene solo manzanas.

`preprocesamiento_<etapa>.csv` (columnas `seccion, clave, medida_continua, operador, valor`): filas `global` (`version`,
`etapa`, `segmentador`, `tam_canvas`, `normalizacion`, `origen_umbrales`, `opciones.tallo|tono|extras|color`) y una fila
`variable` por columna del vector, en su orden, con la medida continua, el operador (`>=`/`<=`) y el umbral **exacto**.
`CarGross.py --guardar-modelo` lo toma automaticamente si esta junto al dataset (o con `--preprocesamiento`) y lo valida.

## manifiesto_imagenes.csv

`foto_id`, `ruta` (relativa a `datasets/processed_images_fruits/`), `estrato` (origen), `tipo_real`, `calidad_real` (solo
manzanas), `marca_tiempo` (del nombre de archivo), `grupo`, `bloque`, `id_orig_morfologia`, `id_orig_calidad`.

* **`grupo`**: fotos de la misma carpeta con <= 3 s de diferencia (o contador de camara consecutivo): **proxy** de
  "misma fruta/rafaga". **No existe** id de fruta fisica ni de sesion; el proxy reduce la fuga entre particiones pero no
  demuestra independencia.
* **`bloque`**: `informe` = 30 % de los grupos reservado (semilla 8, estratificado por origen); `desarrollo` = el resto. Las
  339 fotos ya se habian inspeccionado antes: el informe **no es un conjunto externo intacto**.
