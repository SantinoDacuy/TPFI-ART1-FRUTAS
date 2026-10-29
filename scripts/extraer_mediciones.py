"""
scripts/extraer_mediciones.py
Calcula las mediciones continuas de TODAS las fotos del banco con cada
segmentador (v1 del primer avance y v2) y las guarda en CSV:

    resultados/mediciones/mediciones_v1.csv
    resultados/mediciones/mediciones_v2.csv

Es la unica parte del flujo que necesita las fotos (la carpeta de imagenes esta
fuera del repositorio por su tamano). El resto de los experimentos
(scripts/experimentos.py) trabaja sobre estos CSV, asi que se puede reproducir
sin las fotos. El procesamiento es determinista (GrabCut con semilla fija).

Uso:
    python scripts/extraer_mediciones.py                 # ambos segmentadores
    python scripts/extraer_mediciones.py --segmentador v2 --procesos 4
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import configuracion as cfg
from validacion import COD_ENTRADA, COD_OK

CAMPOS_SEG = ["seg_candidato", "seg_puntaje", "seg_advertencia"]
CAMPOS_FONDO = ["fondo_L", "fondo_a", "fondo_b", "fondo_saturacion", "fondo_verde"]


def _fondo(args):
    """Descriptores del ENTORNO de la foto (marco exterior del 5%), no de la fruta. Sirven solo para el
    diagnostico de confusion fondo-clase (experimentos.py fondo); no entran a ninguna red."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    ruta, fid = args
    img = cv2.imread(ruta)
    h, w = img.shape[:2]
    mh, mw = max(2, h // 20), max(2, w // 20)
    marco = np.zeros((h, w), bool)
    marco[:mh], marco[-mh:], marco[:, :mw], marco[:, -mw:] = True, True, True, True
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)[marco]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[marco]
    verde = np.mean((hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 60))
    return fid, {"fondo_L": lab[:, 0].mean(), "fondo_a": lab[:, 1].mean(), "fondo_b": lab[:, 2].mean(),
                 "fondo_saturacion": hsv[:, 1].mean() / 255.0, "fondo_verde": float(verde)}


def _medir(args):
    import cv2
    cv2.setNumThreads(1)
    import extraccion
    ruta, fid, seg, norm = args
    img = cv2.imread(ruta)
    if img is None:
        return fid, None, f"no se pudo leer {ruta}"
    try:
        m = extraccion.medir_foto(img, segmentador=seg, normalizacion=norm)
    except Exception as e:  # noqa: BLE001 - se informa por foto, no se corta el lote
        return fid, None, f"{type(e).__name__}: {e}"
    info = m.info_seg
    p = (info.get("puntajes") or {}).get(info.get("candidato"), "")
    return fid, {**m.crudos, "seg_candidato": info.get("candidato", ""), "seg_puntaje": p,
                 "seg_advertencia": info.get("advertencia") or ""}, None


def main():
    ap = argparse.ArgumentParser(description="Mediciones continuas de todas las fotos del banco")
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES, help="raiz de las imagenes")
    ap.add_argument("--salida", default=os.path.join(cfg.DIR_RESULTADOS, "mediciones"))
    ap.add_argument("--segmentador", choices=("v1", "v2", "ambos"), default="ambos")
    ap.add_argument("--normalizaciones", nargs="+", default=["ninguna", "estirar", "gris_mundo", "gamma_media"],
                    help="normalizaciones de iluminacion a medir con el segmentador v1 (v2 solo 'ninguna')")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    from evaluacion import banco
    try:
        fotos = banco.inventariar(a.imagenes)
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nAyuda: python scripts/extraer_mediciones.py --help", file=sys.stderr)
        return COD_ENTRADA
    os.makedirs(a.salida, exist_ok=True)
    combos = [(seg, norm) for seg in (("v1", "v2") if a.segmentador == "ambos" else (a.segmentador,))
              for norm in (a.normalizaciones if seg == "v1" else ["ninguna"])]
    for seg, norm in combos:
        t0 = time.time()
        tareas = [(os.path.join(a.imagenes, *f["ruta"].split("/")), f["foto_id"], seg, norm) for f in fotos]
        with ProcessPoolExecutor(max_workers=a.procesos) as ex:
            res = list(ex.map(_medir, tareas, chunksize=4))
        errores = [(fid, e) for fid, v, e in res if v is None]
        filas = [(fid, v) for fid, v, _ in res if v is not None]
        ruta = os.path.join(a.salida, f"mediciones_{seg}.csv" if norm == "ninguna" else f"mediciones_{seg}_{norm}.csv")
        campos = ["foto_id"] + sorted(k for k in filas[0][1] if k not in CAMPOS_SEG) + CAMPOS_SEG
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for fid, v in filas:
                w.writerow({"foto_id": fid, **{k: repr(x) if isinstance(x, float) else x for k, x in v.items()}})
        print(f"[{seg}/{norm}] {len(filas)}/{len(fotos)} fotos en {time.time() - t0:.0f}s -> {ruta}")
        for fid, e in errores:
            print(f"  ERROR {fid}: {e}")
        if errores:
            return 1
    tareas = [(os.path.join(a.imagenes, *f["ruta"].split("/")), f["foto_id"]) for f in fotos]
    with ProcessPoolExecutor(max_workers=a.procesos) as ex:
        fondo = list(ex.map(_fondo, tareas, chunksize=8))
    ruta = os.path.join(a.salida, "fondo.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto_id"] + CAMPOS_FONDO)
        for fid, v in fondo:
            w.writerow([fid] + [repr(float(v[c])) for c in CAMPOS_FONDO])
    print(f"[fondo] {len(fondo)} fotos -> {ruta}")
    return COD_OK


if __name__ == "__main__":
    sys.exit(main())
