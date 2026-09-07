# TFI ART1 — Clasificación de frutas (Grupo 2)

Sistema de clasificación automatizada de frutas mediante red de
Carpenter-Grossberg (ART1). Materia Redes Neuronales, UADER FCyT.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura

```
preprocesador.py               # pipeline OpenCV: segmentación, centrado/
                                # escalado, extracción de features, binarización
scripts/
  generar_dataset_morfologia.py  # arma datasets/dataset_morfologia.csv
                                  # desde fotos locales de Fruits-360
CarGross.py                    # red ART1 (PENDIENTE de implementar)
datasets/
  dataset_morfologia.csv               # 150 filas, fotos reales (Fruits-360)
  dataset_morfologia_sintetico_pruebas.csv  # patrones limpios, solo para
                                              # testear la matemática de ART1
  dataset_calidad_sintetico_pruebas.csv     # ídem, calidad de manzanas
comparacion_pipeline.png       # ejemplo visual: original / máscara / centrado
```

## Estado actual

- [x] Justificación del problema y arquitectura de dos etapas (confirmada
      por la cátedra)
- [x] Preprocesamiento con OpenCV (segmentación, centrado+escalado, features
      en grises + color) — probado sobre 150 fotos reales, 0 errores
- [x] `dataset_morfologia.csv` real (Cítricos / Tropicales / Carozo)
- [ ] `dataset_calidad.csv` real — pendiente: bajar dataset de calidad de
      manzanas (Kaggle: "Fresh and Stale Classification" o "Fruit Quality
      Classification") y correr un script equivalente. Pendiente también
      resolver cómo medir `es_grande` / `es_pesada` sin referencia de escala
      en la foto.
- [ ] `CarGross.py` — matemática de ART1 (ver docstring del archivo)
- [ ] `informe_corridas.pdf`, `manual_referencia.pdf`, PPT con sonido

## Referencias

[1] Sánchez-Sinencio, E. & Lau, C. (1992). Artificial Neural Networks. IEEE Press.
