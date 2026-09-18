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
preprocesador.py               # pipeline OpenCV hibrido: segmentacion, centrado/
                                # escalado, extraccion de features, binarizacion
preprocesador_calidad.py       # pipeline y umbrales de referencia de calidad
CarGross.py                    # red ART1: clasificacion, persistencia e inferencia
probar_imagen.py               # inferencia interactiva en vivo (dos etapas + grafico)
demo.py                        # ejecucion automatizada de punta a punta
scripts/
  curar_dataset.py             # curacion exhaustiva y generacion de ambos datasets
datasets/
  dataset_morfologia.csv       # 180 filas reales (60 citricos, 60 tropicales, 60 carozo)
  dataset_calidad_real.csv     # 180 manzanas reales (60 premium, 60 comercial, 60 descarte)
modelos/
  art1_morfologia.json         # modelo ART1 Etapa 1 entrenado y persistido (rho=0.85)
  art1_calidad.json            # modelo ART1 Etapa 2 entrenado y persistido (rho=0.70)
```

## Uso

### 1. Demostracion Completa
Ejecuta las corridas de ambas etapas con distintos valores de vigilancia ($\rho$):
```bash
python demo.py --solo-corridas
```

### 2. Prueba en Vivo con una Imagen Individual (Defensa Oral)
Evalua una fotografia cualquiera en dos etapas (Tipo de fruta $\to$ Calidad de manzana) y genera un panel visual con el preprocesamiento (`diagnostico_clasificacion.png`).

Ejemplos incluidos listos para probar:
```bash
# Manzana de buena calidad (resona en Etapa 1 y pasa a Etapa 2)
python probar_imagen.py --imagen ejemplos/manzana_premium.jpg

# Manzana con dano / descarte
python probar_imagen.py --imagen ejemplos/manzana_comercial_defecto.jpg

# Citrico (resona en Morfologia y omite Etapa 2 de manzana)
python probar_imagen.py --imagen ejemplos/naranja.jpg

# Deteccion de novedad con plasticidad activa (aprende una categoria nueva)
python probar_imagen.py --imagen ejemplos/manzana_descarte_severo.jpg --aprender
```


## Estado del Proyecto

- [x] Justificacion del problema y arquitectura de dos etapas (aprobada por catedra).
- [x] Preprocesamiento OpenCV robusto (segmentacion hibrida HSV/Otsu, centrado $200\times200$, metadato de color).
- [x] Datasets reales (150 frutas en Morfologia y 90 manzanas en Calidad).
- [x] `CarGross.py` implementado segun Carpenter-Grossberg (1987) con persistencia, inferencia sin mutacion de pesos y soporte de vector nulo.
- [x] Script de prueba interactiva en vivo con reporte grafico (`probar_imagen.py`).
- [x] Demo automatizado (`demo.py`).


## Referencias

[1] Sánchez-Sinencio, E. & Lau, C. (1992). Artificial Neural Networks. IEEE Press.
