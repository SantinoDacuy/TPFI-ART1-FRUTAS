"""
scripts/figura_segmentacion.py
Figura para el informe: contorno de la segmentacion v1 (verde) y v2 (rojo) sobre
la misma foto, con 3 fotos al azar (semilla fija) por origen del banco.

Uso:  python scripts/figura_segmentacion.py [--salida docs/figuras/segmentacion_v1_v2.png]
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import configuracion as cfg
import preprocesador as P
from evaluacion import banco


def main():
    ap = argparse.ArgumentParser(description="Figura comparativa de segmentadores")
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "docs", "figuras", "segmentacion_v1_v2.png"))
    ap.add_argument("--por-origen", type=int, default=3)
    a = ap.parse_args()
    fotos = banco.inventariar(a.imagenes)
    rng = random.Random(11)
    filas = []
    for est in ("manzana_good", "manzana_mixed", "manzana_bad", "naranja", "banana"):
        sel = rng.sample([f for f in fotos if f["estrato"] == est], a.por_origen)
        fila = []
        for f in sel:
            img = cv2.imread(os.path.join(a.imagenes, *f["ruta"].split("/")))
            img = cv2.resize(img, (170, 170))
            vis = img.copy()
            for mascara, color in ((P.segmentar(img), (0, 200, 0)), (P.segmentar_v2(img), (0, 0, 255))):
                cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cont, -1, color, 2)
            cv2.putText(vis, f["foto_id"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            fila.append(vis)
        filas.append(np.hstack(fila))
    lienzo = np.vstack(filas)
    banda = np.full((26, lienzo.shape[1], 3), 255, np.uint8)
    cv2.putText(banda, "verde: v1 (primer avance) | rojo: v2 | filas: good, mixed, bad, naranja, banana",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    cv2.imwrite(a.salida, np.vstack([banda, lienzo]))
    print(f"figura guardada en {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
