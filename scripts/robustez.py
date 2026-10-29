"""
scripts/robustez.py
Diagnostico de dependencia de fondo e iluminacion: se perturban las FOTOS (no los modelos) y se mide cuanto
cambian las predicciones de la cadena de dos etapas.

  * Fondo: la fruta (mascara de GrabCut, borde suavizado) se pega sobre un fondo liso (blanco, gris, verde, azul).
    Si el sistema mirara la fruta, la prediccion no deberia cambiar mucho; si mira el escenario, si.
  * Iluminacion: gamma (mas oscura / mas clara), ganancia y tinte calido / frio aplicados a toda la foto.

Para cada condicion se informa, sobre las fotos elegidas (por defecto el bloque de INFORME del manifiesto):
  - prediccion estable: la cadena (tipo y calidad) coincide con la de la foto original;
  - exactitud del tipo y de la cadena contra las etiquetas de referencia;
  - tasa de derivacion a calidad.
Es un DIAGNOSTICO: las mascaras son aproximadas y la sustitucion de fondo genera imagenes que el banco no
tiene; no corrige nada ni prueba causalidad. Requiere las fotos (fuera del repositorio).

Uso:
    python scripts/robustez.py --morfologia modelos_v2/principal/art1_morfologia.json \
        --calidad modelos_v2/principal/art1_calidad.json [--sistema-legado] [--salida resultados/ronda2]
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import configuracion as cfg
import inferencia
import preprocesador as P
from CarGross import cargar_modelo
from evaluacion import banco
from validacion import ALIAS_MANZANA, COD_ENTRADA, COD_OK, ErrorEntrada

FONDOS = {"fondo_blanco": (255, 255, 255), "fondo_gris": (128, 128, 128), "fondo_verde": (50, 140, 60),
          "fondo_azul": (170, 110, 40)}


def _gamma(img, g):
    return (255.0 * (img / 255.0) ** g).clip(0, 255).astype(np.uint8)


ILUMINACION = {
    "luz_oscura": lambda im: _gamma(im, 1.6),
    "luz_clara": lambda im: _gamma(im, 0.6),
    "ganancia_baja": lambda im: np.clip(im.astype(np.float32) * 0.7, 0, 255).astype(np.uint8),
    "ganancia_alta": lambda im: np.clip(im.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
    "tinte_calido": lambda im: np.clip(im.astype(np.float32) * np.array([0.85, 1.0, 1.15]), 0, 255).astype(np.uint8),
    "tinte_frio": lambda im: np.clip(im.astype(np.float32) * np.array([1.15, 1.0, 0.85]), 0, 255).astype(np.uint8),
}


def sustituir_fondo(img, mascara, color):
    suave = cv2.GaussianBlur((mascara > 0).astype(np.float32), (0, 0), 1.5)[..., None]
    fondo = np.full_like(img, color)
    return (img.astype(np.float32) * suave + fondo.astype(np.float32) * (1 - suave)).astype(np.uint8)


def condiciones(img):
    """Imagenes perturbadas de una foto: {nombre: imagen}."""
    mascara = P.segmentar_v2(img)
    out = {"original": img}
    for n, c in FONDOS.items():
        out[n] = sustituir_fondo(img, mascara, c)
    for n, f in ILUMINACION.items():
        out[n] = f(img)
    return out


def predecir(img, m_morf, m_cal):
    try:
        r = inferencia.inferir_imagen(img, m_morf, m_cal)
    except ErrorEntrada:
        return ("(sin fruta)", None, False)
    e2 = r["etapa2"]
    return (r["etapa1"]["etiqueta"], e2["etiqueta"] if e2 else None, r["derivada"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robustez al fondo y a la iluminacion (diagnostico)")
    ap.add_argument("--morfologia", help="modelo de la etapa 1 (formato 2, de fotografias)")
    ap.add_argument("--calidad", help="modelo de la etapa 2")
    ap.add_argument("--legado", action="store_true", help="usa los modelos entregados en modelos/")
    ap.add_argument("--manifiesto", default=os.path.join(cfg.DIR_DATASETS_V2, "manifiesto_imagenes.csv"))
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    ap.add_argument("--bloque", default="informe", choices=("informe", "desarrollo", "todas"))
    ap.add_argument("--salida", default=os.path.join(cfg.DIR_RESULTADOS, "ronda2"))
    ap.add_argument("--nombre", default=None, help="nombre del sistema en la tabla")
    a = ap.parse_args(argv)
    try:
        if a.legado:
            rm = os.path.join(cfg.DIR_MODELOS_ORIGINAL, "art1_morfologia.json")
            rc = os.path.join(cfg.DIR_MODELOS_ORIGINAL, "art1_calidad.json")
        else:
            if not (a.morfologia and a.calidad):
                raise ErrorEntrada("indique --morfologia y --calidad (o --legado)")
            rm, rc = a.morfologia, a.calidad
        m_morf, m_cal = cargar_modelo(rm), cargar_modelo(rc)
        fotos = banco.leer_manifiesto(a.manifiesto)
    except (ErrorEntrada, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}\nAyuda: python scripts/robustez.py --help", file=sys.stderr)
        return COD_ENTRADA
    if a.bloque != "todas":
        fotos = [f for f in fotos if f.get("bloque") == a.bloque]
    nombre = a.nombre or ("original entregado" if a.legado else os.path.basename(os.path.dirname(os.path.abspath(rm))))
    registros = []
    for f in fotos:
        ruta = os.path.join(a.imagenes, *f["ruta"].split("/"))
        img = cv2.imread(ruta)
        if img is None:
            print(f"ERROR: no se pudo leer {ruta}", file=sys.stderr)
            return COD_ENTRADA
        base = None
        for cond, im in condiciones(img).items():
            pred = predecir(im, m_morf, m_cal)
            if cond == "original":
                base = pred
            es_manz = f["tipo_real"] == ALIAS_MANZANA
            tipo_ok = pred[0] == f["tipo_real"]
            cadena_ok = tipo_ok and (pred[1] == f["calidad_real"] if es_manz else True)
            registros.append({"sistema": nombre, "foto_id": f["foto_id"], "condicion": cond, "tipo_real": f["tipo_real"],
                              "calidad_real": f["calidad_real"], "tipo_pred": pred[0] or "", "calidad_pred": pred[1] or "",
                              "derivada": int(pred[2]), "estable": int(pred[:2] == base[:2]),
                              "tipo_ok": int(tipo_ok), "cadena_ok": int(cadena_ok)})
    os.makedirs(a.salida, exist_ok=True)
    ruta_csv = os.path.join(a.salida, f"robustez_{nombre.replace(' ', '_')}.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(registros[0]))
        w.writeheader()
        w.writerows(registros)
    L = [f"### {nombre} ({len(fotos)} fotos, bloque '{a.bloque}')\n",
         "| condicion | prediccion estable | tipo correcto | cadena correcta | derivadas a calidad |", "|---|---|---|---|---|"]
    for cond in ["original"] + list(FONDOS) + list(ILUMINACION):
        r = [x for x in registros if x["condicion"] == cond]
        n = len(r)
        L.append(f"| {cond} | {100 * sum(x['estable'] for x in r) / n:.1f}% | {100 * sum(x['tipo_ok'] for x in r) / n:.1f}% | "
                 f"{100 * sum(x['cadena_ok'] for x in r) / n:.1f}% | {sum(x['derivada'] for x in r)}/{n} |")
    texto = "\n".join(L) + "\n"
    with open(os.path.join(a.salida, f"robustez_{nombre.replace(' ', '_')}.md"), "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(texto)
    return COD_OK


if __name__ == "__main__":
    sys.exit(main())
