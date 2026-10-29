"""
scripts/trazar_ids_originales.py
Reconstruye la correspondencia entre los identificadores de los CSV ENTREGADOS
(fruta_id: FM054, PRM016, ...) y las fotos del banco, volviendo a ejecutar la
seleccion del curador del primer avance sobre las fotos que quedan.

Por que es posible: el curador recorria cada carpeta en orden alfabetico y se
quedaba con las primeras 60 fotos que pasaban sus filtros; las fotos descartadas
se borraron, pero las que quedan siguen pasando los mismos filtros, asi que la
misma recorrida elige las mismas. Se VERIFICA comparando, fila por fila, el
vector binario y la etiqueta contra los CSV entregados (180/180 en cada uno).

Salida: datasets_v2/trazabilidad_ids_originales.csv
    fruta_id_original, dataset, ruta, verificada

Uso:  python scripts/trazar_ids_originales.py
"""
import csv
import json
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RAIZ, "baseline_original")
IMAGENES = os.path.join(RAIZ, "datasets", "processed_images_fruits")


def _worker():
    sys.path.insert(0, BASE)
    sys.path.insert(1, os.path.join(BASE, "scripts"))
    import curar_dataset as cd                                  # del snapshot
    from preprocesador import binarizar
    from preprocesador_calidad import procesar_lote_calidad
    cd.BASE_IMGS = IMAGENES

    def elegir(carpeta, evaluar, clase):
        out = []
        for f in sorted(os.listdir(carpeta)):
            if len(out) >= cd.OBJETIVO_POR_CLASE or not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            p = os.path.join(carpeta, f)
            crudos, _ = evaluar(p, clase)
            if crudos is not None:
                out.append((p, crudos))
        return out

    j = lambda *a: os.path.join(IMAGENES, *a)
    ag, am, ab = j("Good Quality_Fruits", "Apple_Good"), j("Mixed Qualit_Fruits", "Apple"), j("Bad Quality_Fruits", "Apple_Bad")
    og, bg = j("Good Quality_Fruits", "Orange_Good"), j("Good Quality_Fruits", "Banana_Good")
    morf = {"carozo": elegir(ag, cd.evaluar_imagen_morfologia, "carozo"),
            "citrico": elegir(og, cd.evaluar_imagen_morfologia, "citrico"),
            "tropical": elegir(bg, cd.evaluar_imagen_morfologia, "tropical")}
    cal = {"premium": elegir(ag, cd.evaluar_imagen_calidad, "premium"),
           "comercial": elegir(am, cd.evaluar_imagen_calidad, "comercial"),
           "descarte": elegir(ab, cd.evaluar_imagen_calidad, "descarte")}

    cols_m = ["es_esferica", "cascara_rugosa", "tiene_tallo", "densidad_alta", "pigmentacion_oscura", "color_uniforme"]
    cols_c = ["es_grande", "es_pesada", "textura_firme", "sin_defectos_superficie", "color_homogeneo"]
    filas = {}
    i = 1
    for tipo, lista in morf.items():
        pref = {"carozo": "FM", "citrico": "FC", "tropical": "FT"}[tipo]
        for p, c in lista:
            b = binarizar(c)
            filas[f"{pref}{i:03d}"] = ("morfologia", p, [b[k] for k in cols_m], tipo)
            i += 1
    rutas = [(p, q) for q, lista in cal.items() for p, _ in lista]
    fb, _, _ = procesar_lote_calidad(rutas)
    pref = {"premium": "PRM", "comercial": "COM", "descarte": "DES"}
    for idx, (f, (p, q)) in enumerate(zip(fb, rutas), 1):
        filas[f"{pref[q]}{idx:03d}"] = ("calidad", p, [f[k] for k in cols_c], q)

    salida = []
    for dataset, nombre, cols, col_et in (("morfologia", "dataset_morfologia.csv", cols_m, "tipo_real"),
                                          ("calidad", "dataset_calidad_real.csv", cols_c, "calidad_real")):
        with open(os.path.join(BASE, "datasets", nombre), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ds, ruta, vec, et = filas[r["fruta_id"]]
                ok = ds == dataset and vec == [int(r[c]) for c in cols] and et == r[col_et]
                salida.append({"fruta_id_original": r["fruta_id"], "dataset": dataset,
                               "ruta": os.path.relpath(ruta, IMAGENES).replace(os.sep, "/"),
                               "verificada": "si" if ok else "NO"})
    json.dump(salida, sys.stdout)


def main():
    if "--worker" in sys.argv:
        _worker()
        return 0
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker"], capture_output=True,
                       text=True, encoding="utf-8")
    if p.returncode != 0:
        print("ERROR al reconstruir la seleccion original:\n" + p.stderr, file=sys.stderr)
        return 4
    filas = json.loads(p.stdout)
    malas = [f for f in filas if f["verificada"] != "si"]
    destino = os.path.join(RAIZ, "datasets_v2")
    os.makedirs(destino, exist_ok=True)
    with open(os.path.join(destino, "trazabilidad_ids_originales.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas) - len(malas)}/{len(filas)} filas verificadas contra los CSV entregados")
    return 0 if not malas else 1


if __name__ == "__main__":
    sys.exit(main())
