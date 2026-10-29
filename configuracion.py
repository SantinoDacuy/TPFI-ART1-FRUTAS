"""
configuracion.py
Unica fuente de parametros por etapa (vigilancia rho, L, tope de categorias,
epocas, politica de vector nulo, desempate de etiquetas) y de rutas.

Los valores viven en config/parametros_etapas.csv (CSV, segun la consigna). El
entrenamiento, la regeneracion de modelos, la demo y las herramientas de
evaluacion leen ese archivo; la INFERENCIA no lo lee: usa siempre los valores
guardados dentro de cada modelo, que son los que se usaron al entrenar.

Columnas del CSV: etapa, rho, L, max_categorias, epocas, politica_nulo,
desempate, fuente (texto libre: de donde salio el valor).
"""
import csv
import os

DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_PARAMETROS = os.path.join(DIR_RAIZ, "config", "parametros_etapas.csv")

# Entregables originales (no se tocan) y resultados nuevos (carpetas distintas)
DIR_DATASETS_ORIGINAL = os.path.join(DIR_RAIZ, "datasets")
DIR_MODELOS_ORIGINAL = os.path.join(DIR_RAIZ, "modelos")
DIR_DATASETS_V2 = os.path.join(DIR_RAIZ, "datasets_v2")
DIR_MODELOS_V2 = os.path.join(DIR_RAIZ, "modelos_v2")
DIR_RESULTADOS = os.path.join(DIR_RAIZ, "resultados")
DIR_IMAGENES = os.path.join(DIR_RAIZ, "datasets", "processed_images_fruits")
DIR_BASELINE = os.path.join(DIR_RAIZ, "baseline_original")

_ETAPAS = ("morfologia", "calidad")


def cargar_parametros(ruta=RUTA_PARAMETROS):
    """Devuelve {etapa: dict} con tipos convertidos. Levanta ValueError con un
    mensaje claro si el archivo es invalido (nunca asume valores en silencio)."""
    if not os.path.isfile(ruta):
        raise ValueError(f"no se encontro el archivo de parametros '{ruta}'")
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    out = {}
    for fila in filas:
        etapa = (fila.get("etapa") or "").strip()
        if etapa not in _ETAPAS:
            raise ValueError(f"{ruta}: etapa desconocida {etapa!r} (esperadas: {_ETAPAS})")
        try:
            out[etapa] = {
                "rho": float(fila["rho"]), "L": float(fila["L"]),
                "max_categorias": int(fila["max_categorias"]),
                "epocas": int(fila["epocas"]),
                "politica_nulo": fila["politica_nulo"].strip(),
                "desempate": fila["desempate"].strip(),
                "fuente": (fila.get("fuente") or "").strip(),
            }
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"{ruta}: fila de '{etapa}' invalida ({e})")
    faltan = [e for e in _ETAPAS if e not in out]
    if faltan:
        raise ValueError(f"{ruta}: faltan las etapas {faltan}")
    return out


def parametros_etapa(etapa, ruta=RUTA_PARAMETROS):
    return cargar_parametros(ruta)[etapa]
