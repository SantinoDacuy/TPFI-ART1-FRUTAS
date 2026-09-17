"""
extractor_calidad.py (CORREGIDO)
Genera datasets/dataset_calidad_real.csv usando preprocesador_calidad.py
(variables de CALIDAD: es_grande, es_pesada, textura_firme,
sin_defectos_superficie, color_homogeneo) - NO usa preprocesador.py
(ese es para morfologia/variedad, son variables distintas).

Mapeo de carpetas del dataset descargado a etiquetas de calidad:
  Good Quality_Fruits/Apple_Good/  -> premium
  Mixed Qualit_Fruits/Apple/       -> comercial   (typo original, no tocar)
  Bad Quality_Fruits/Apple_Bad/    -> descarte

USO:
    python scripts/extractor_calidad.py              # 30 imgs/clase (default)
    python scripts/extractor_calidad.py --n 25
    python scripts/extractor_calidad.py --rng-seed 7
"""
import os
import sys
import csv
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocesador_calidad import procesar_lote_calidad  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "datasets", "processed_images_fruits")
SALIDA = os.path.join(os.path.dirname(__file__), "..", "datasets", "dataset_calidad_real.csv")

CARPETAS = [
    (os.path.join(BASE, "Good Quality_Fruits", "Apple_Good"), "premium",   "PRM"),
    (os.path.join(BASE, "Mixed Qualit_Fruits", "Apple"),      "comercial", "COM"),
    (os.path.join(BASE, "Bad Quality_Fruits",  "Apple_Bad"),  "descarte",  "DES"),
]

COLUMNAS = ["fruta_id", "es_grande", "es_pesada", "textura_firme",
            "sin_defectos_superficie", "color_homogeneo", "calidad_real"]

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset_calidad_real.csv desde imagenes reales de Apple"
    )
    parser.add_argument("--n", type=int, default=30,
                        help="Imagenes a tomar por clase (default 30)")
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad (default 42)")
    args = parser.parse_args()

    random.seed(args.rng_seed)

    print("=" * 62)
    print("EXTRACTOR DE FEATURES DE CALIDAD — manzanas reales")
    print("=" * 62)
    print(f"  Variables    : es_grande, es_pesada, textura_firme,")
    print(f"                 sin_defectos_superficie, color_homogeneo")
    print(f"  Binarizacion : por mediana del lote completo (no umbral fijo)")
    print(f"  Imgs/clase   : {args.n}  |  Semilla: {args.rng_seed}")
    print("-" * 62)

    rutas_con_etiqueta = []
    ids = []

    for carpeta, etiqueta, prefijo in CARPETAS:
        if not os.path.isdir(carpeta):
            print(f"  [AVISO] No existe '{carpeta}', se salta esa clase.")
            continue
        archivos = sorted(
            f for f in os.listdir(carpeta)
            if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
        )
        if not archivos:
            print(f"  [AVISO] Sin imagenes en '{carpeta}', se salta.")
            continue
        elegidos = random.sample(archivos, min(args.n, len(archivos)))
        for i, a in enumerate(elegidos, 1):
            rutas_con_etiqueta.append((os.path.join(carpeta, a), etiqueta))
            ids.append(f"{prefijo}{i:03d}")

    if not rutas_con_etiqueta:
        raise SystemExit("ERROR: no se encontraron imagenes en ninguna carpeta.")

    filas, crudos, medianas = procesar_lote_calidad(rutas_con_etiqueta)

    # Si hubo errores en el lote, filas tiene menos elementos que ids -> recortar
    ids = ids[:len(filas)]

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        for fid, fila in zip(ids, filas):
            w.writerow({"fruta_id": fid, **fila})

    print(f"\n  OK: {len(filas)} filas escritas en:")
    print(f"      {SALIDA}")

    # Resumen de metricas crudas por clase (para el informe)
    print("\n" + "=" * 62)
    print("RESUMEN — promedios de metricas crudas por clase")
    print("(antes de binarizar — utiles para el informe de corridas)")
    print("=" * 62)
    por_clase = defaultdict(list)
    for c, fila in zip(crudos, filas):
        por_clase[fila["calidad_real"]].append(c)

    for etiqueta in ["premium", "comercial", "descarte"]:
        grupo = por_clase.get(etiqueta, [])
        if not grupo:
            continue
        prom = {k: round(sum(g[k] for g in grupo) / len(grupo), 1)
                for k in grupo[0]}
        print(f"\n  {etiqueta.upper():10} (n={len(grupo)}):")
        for k, v in prom.items():
            print(f"    {k:<25} {v}")

    print("\n" + "=" * 62)
    print("MEDIANAS DEL LOTE usadas como umbral de binarizacion:")
    for k, v in medianas.items():
        print(f"  {k:<25} {round(v, 1)}")

    # Verificacion final contra consigna
    n_vars = len(COLUMNAS) - 2  # saca fruta_id y calidad_real
    print("\n" + "-" * 62)
    print(f"  Filas totales   : {len(filas)}  (consigna >= 50: {'SI' if len(filas)>=50 else 'NO'})")
    print(f"  Vars binarias   : {n_vars}   (consigna >= 5:  {'SI' if n_vars>=5 else 'NO'})")
    print("=" * 62)


if __name__ == "__main__":
    main()
