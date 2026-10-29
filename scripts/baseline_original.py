"""
scripts/baseline_original.py
Linea de base reproducible: ejecuta el codigo, los datasets y los modelos del
primer avance TAL COMO FUERON ENTREGADOS (snapshot en baseline_original/, commit
f01b4dd) y comprueba los hallazgos observados antes de cambiar nada.

Uso:
    python scripts/baseline_original.py                 # escribe resultados/baseline_original.*
    python scripts/baseline_original.py --sin-imagenes  # solo la parte de CSV (no requiere fotos)

Salida: resultados/baseline_original.json (todo), resultados/baseline_original.md
(resumen legible) y resultados/baseline_fotos.csv (una fila por foto).

El trabajo que importa codigo del snapshot corre en un subproceso (--worker)
para no mezclar los modulos del primer avance con los actuales, que tienen los
mismos nombres (CarGross, preprocesador, ...).
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(RAIZ, "baseline_original")
COMMIT_BASE = "f01b4dd"

# Cifras que traia el pedido, para contrastar lo que se reproduce
ESPERADO_CORRIDAS = {
    ("morfologia", 0.70): (14, 76.11), ("morfologia", 0.85): (19, 75.56),
    ("calidad", 0.70): (17, 81.11), ("calidad", 0.85): (21, 81.67), ("calidad", 0.95): (21, 81.67),
}
ESPERADO_FOTOS = {"tipo_correcto": 167, "n_fotos": 339, "manzanas_derivadas": 96, "n_manzanas": 219,
                  "cadena_completa": 65, "no_manzanas_derivadas": 44, "n_no_manzanas": 120}


# ---------------------------------------------------------------------------
# WORKER: corre con el codigo del primer avance
# ---------------------------------------------------------------------------
def _worker(sin_imagenes):
    sys.path.insert(0, BASE)
    sys.path.insert(1, os.path.join(BASE, "scripts"))
    sys.path.append(RAIZ)  # al final: 'evaluacion' es unico; CarGross etc. salen del snapshot
    import csv
    import tempfile

    import CarGross as cg                                   # del snapshot
    out = {"corridas": [], "datasets": {}, "hallazgos": {}}

    csvs = {"morfologia": os.path.join(BASE, "datasets", "dataset_morfologia.csv"),
            "calidad": os.path.join(BASE, "datasets", "dataset_calidad_real.csv")}
    modelos = {"morfologia": os.path.join(BASE, "modelos", "art1_morfologia.json"),
               "calidad": os.path.join(BASE, "modelos", "art1_calidad.json")}

    # --- A. integridad de los datasets -------------------------------------
    for etapa, ruta in csvs.items():
        ids, X, meta, cf, cm = cg.leer_csv(ruta)
        et = [m[cm[0]] for m in meta]
        por_vec = {}
        for x, e in zip(X, et):
            por_vec.setdefault(tuple(x), Counter())[e] += 1
        mezclados = {v: c for v, c in por_vec.items() if len(c) > 1}
        techo = sum(max(c.values()) for c in por_vec.values()) / len(X)
        out["datasets"][etapa] = {
            "filas": len(X), "n_features": len(cf), "features": cf, "ids_unicos": len(set(ids)) == len(ids),
            "clases": dict(Counter(et)), "vectores_nulos": sum(1 for x in X if not any(x)),
            "vectores_distintos": len(por_vec), "vectores_con_etiquetas_mezcladas": len(mezclados),
            "muestras_en_vectores_mezclados": sum(sum(c.values()) for c in mezclados.values()),
            "techo_de_pureza_por_vectores_identicos": round(techo, 4),
            "columnas_constantes": [c for i, c in enumerate(cf) if len({x[i] for x in X}) == 1],
        }

    # --- B. corridas de referencia (una pasada, orden original) -----------------
    for (etapa, rho), (k_esp, p_esp) in ESPERADO_CORRIDAS.items():
        r = cg.ejecutar(csvs[etapa], rho)
        asg, col = r["asignaciones"], r["cols_metadata"][0]
        total = 0
        for c in set(asg):
            m = Counter(r["metadata"][i][col] for i, a in enumerate(asg) if a == c)
            total += m.most_common(1)[0][1]
        out["corridas"].append({"etapa": etapa, "rho": rho, "clusters": r["n_categorias"],
                                "pureza_online_pct": round(100 * total / len(asg), 2),
                                "esperado_clusters": k_esp, "esperado_pureza_pct": p_esp})

    # --- C. discrepancia entre mapeo online y prototipos finales -----------------
    disc = {}
    for etapa in csvs:
        red, meta = cg.RedART1.cargar(modelos[etapa])
        ids, X, md, cf, cm = cg.leer_csv(csvs[etapa])
        et = [m[cm[0]] for m in md]
        assert cf == meta["cols_features"]
        # el modelo entregado ¿coincide con reentrenar su CSV con su propio rho?
        red2, meta2 = cg.entrenar_modelo(csvs[etapa], red.rho)
        igual = red2.top_down == red.top_down
        # reevaluar cada muestra con los prototipos finales
        sin_res, cambia_cat = 0, 0
        red3 = cg.RedART1(len(cf), red.rho)
        online = [red3.clasificar(x) for x in X]
        por_final = {}
        for i, x in enumerate(X):
            ev = red.evaluar(x)
            if not ev["resuena"]:
                sin_res += 1
            else:
                por_final.setdefault(ev["ganador"], []).append(et[i])
                cambia_cat += ev["ganador"] != online[i]
        etiquetas_cambian = 0
        for cid, info in meta["mapeo_clusters"].items():
            if int(cid) in por_final:
                if Counter(por_final[int(cid)]).most_common(1)[0][0] != info["etiqueta"]:
                    etiquetas_cambian += 1
            else:
                etiquetas_cambian += 1  # categoria sin muestras al reevaluar
        disc[etapa] = {
            "rho_modelo_entregado": red.rho, "categorias": len(red.top_down),
            "modelo_igual_a_reentrenar_su_csv": igual,
            "muestras_sin_resonancia_con_prototipos_finales": sin_res,
            "muestras_cuya_categoria_cambia_al_reevaluar": cambia_cat,
            "n_muestras": len(X),
            "categorias_cuya_etiqueta_cambia_o_queda_vacia": etiquetas_cambian,
            "categorias_sin_muestras_al_reevaluar": len(red.top_down) - len(por_final),
        }
    out["hallazgos"]["mapeo_online_vs_prototipos_finales"] = disc

    # --- D. fallos de CLI / validacion del primer avance -------------------------
    with tempfile.TemporaryDirectory() as tmp:
        malo = os.path.join(tmp, "malo.csv")
        with open(malo, "w", newline="") as f:
            f.write("fruta_id,a,b,c,tipo_real\nX1,1,0,2,t\nX2,0,1,1,t\nX3,1,1,0,u\n")
        ids, X, md, cf, cm = cg.leer_csv(malo)
        out["hallazgos"]["valor_invalido_pasa_a_metadato"] = {
            "features_detectadas": cf, "metadatos_detectados": cm,
            "nota": "la columna c tenia un '2' y paso a metadato; la red se entreno con 2 variables"}
        cli = [sys.executable, os.path.join(BASE, "CarGross.py"), "--csv", csvs["morfologia"],
               "--rho", "0.95", "--max-categorias", "5"]
        p = subprocess.run(cli, capture_output=True, text=True)
        out["hallazgos"]["corrida_infructuosa_codigo_de_salida"] = {
            "comando": "CarGross.py --rho 0.95 --max-categorias 5", "codigo": p.returncode,
            "stdout": p.stdout.strip().splitlines()[:2]}

    # --- E. constantes de regeneracion y umbrales --------------------------------
    def _buscar(archivo, patron):
        with open(os.path.join(BASE, archivo), encoding="utf-8") as f:
            return [f"{archivo}:{i}: {l.strip()}" for i, l in enumerate(f, 1) if re.search(patron, l)]
    out["hallazgos"]["rho_en_rutas_de_regeneracion"] = {
        "modelo_entregado_morfologia": modelos["morfologia"] and json.load(open(modelos["morfologia"]))["rho"],
        "lineas": _buscar("probar_imagen.py", r"rho_morf|rho_cal") + _buscar("scripts/curar_dataset.py", r"entrenar_modelo")}

    if sin_imagenes:
        return out

    # --- F. evaluacion fotografica con los modelos entregados -----------------------
    import cv2
    from preprocesador import binarizar, centrar_y_escalar, extraer_features, segmentar
    from preprocesador_calidad import UMBRALES_REFERENCIA_CALIDAD, binarizar_calidad, extraer_crudos_calidad
    from evaluacion import banco, metricas
    raiz_img = os.path.join(RAIZ, "datasets", "processed_images_fruits")
    fotos = banco.inventariar(raiz_img)
    rm, mm = cg.RedART1.cargar(modelos["morfologia"])
    rc, mc = cg.RedART1.cargar(modelos["calidad"])
    registros = []
    for f in fotos:
        img = cv2.imread(os.path.join(raiz_img, *f["ruta"].split("/")))
        m = segmentar(img)
        ic, mcnt = centrar_y_escalar(img, m)
        b = binarizar(extraer_features(ic, mcnt))
        e = rm.evaluar([b[c] for c in mm["cols_features"]])
        et = mm["mapeo_clusters"][str(e["ganador"])]["etiqueta"] if e["resuena"] else None
        reg = {"foto_id": f["foto_id"], "ruta": f["ruta"], "estrato": f["estrato"],
               "tipo_real": f["tipo_real"], "calidad_real": f["calidad_real"],
               "morf_cluster": e["ganador"], "morf_match": round(e["ratio"], 4), "morf_etiqueta": et,
               "derivada": bool(et and "carozo" in et), "cal_cluster": None, "cal_match": None,
               "cal_etiqueta": None}
        if reg["derivada"]:
            bc = binarizar_calidad(extraer_crudos_calidad(img, m, ic, mcnt), UMBRALES_REFERENCIA_CALIDAD)
            ec = rc.evaluar([bc[c] for c in mc["cols_features"]])
            reg.update(cal_cluster=ec["ganador"], cal_match=round(ec["ratio"], 4),
                       cal_etiqueta=mc["mapeo_clusters"][str(ec["ganador"])]["etiqueta"] if ec["resuena"] else None)
        registros.append(reg)
    out["fotos"] = registros
    return out


# ---------------------------------------------------------------------------
# Resumen (proceso principal, sin importar el snapshot)
# ---------------------------------------------------------------------------
def _resumen_fotos(regs):
    sys.path.insert(0, RAIZ)
    from evaluacion import metricas
    clases = ["citrico", "tropical", "carozo"]
    verdad = [r["tipo_real"] for r in regs]
    pred = [r["morf_etiqueta"] for r in regs]
    manz = [r for r in regs if r["tipo_real"] == "carozo"]
    nom = [r for r in regs if r["tipo_real"] != "carozo"]
    res = {
        "n_fotos": len(regs), "tipo_correcto": sum(v == p for v, p in zip(verdad, pred)),
        "n_manzanas": len(manz), "manzanas_derivadas": sum(r["derivada"] for r in manz),
        "cadena_completa": sum(r["derivada"] and r["cal_etiqueta"] == r["calidad_real"] for r in manz),
        "n_no_manzanas": len(nom), "no_manzanas_derivadas": sum(r["derivada"] for r in nom),
        "sin_resonancia_etapa1": sum(r["morf_cluster"] is None for r in regs),
        "exactitud_tipo": metricas.exactitud(verdad, pred),
        "f1_macro_tipo": metricas.f1_macro(verdad, pred, clases),
        "confusion_tipo": metricas.matriz_confusion(verdad, pred, clases),
        "mayoritaria_tipo": metricas.linea_base_mayoritaria(verdad, "carozo", clases),
        "por_estrato": {},
    }
    for est in sorted({r["estrato"] for r in regs}):
        sub = [r for r in regs if r["estrato"] == est]
        res["por_estrato"][est] = {
            "n": len(sub), "tipo_correcto": sum(r["tipo_real"] == r["morf_etiqueta"] for r in sub),
            "derivadas": sum(r["derivada"] for r in sub)}
    cal = [r["calidad_real"] for r in manz if r["derivada"]]
    cp = [r["cal_etiqueta"] for r in manz if r["derivada"]]
    res["calidad_exactitud_sobre_derivadas"] = metricas.exactitud(cal, cp)
    return res


def _escribir_md(out, ruta):
    L = ["# Linea de base reproducible (primer avance tal como fue entregado)\n",
         f"Codigo, datasets y modelos: `baseline_original/` (snapshot del commit `{COMMIT_BASE}`).  ",
         f"Python {platform.python_version()} - {platform.platform()}.\n"]
    L.append("## A. Datasets\n")
    L.append("| etapa | filas | variables | vectores distintos | vectores con etiquetas mezcladas | techo de pureza | vectores nulos | columnas constantes |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e, d in out["datasets"].items():
        L.append(f"| {e} | {d['filas']} | {d['n_features']} | {d['vectores_distintos']} | "
                 f"{d['vectores_con_etiquetas_mezcladas']} ({d['muestras_en_vectores_mezclados']} muestras) | "
                 f"{100 * d['techo_de_pureza_por_vectores_identicos']:.2f}% | {d['vectores_nulos']} | "
                 f"{', '.join(d['columnas_constantes']) or '-'} |")
    L.append("\n*Techo de pureza*: maxima pureza alcanzable por CUALQUIER agrupamiento de estos vectores "
             "(vectores identicos con etiquetas distintas no se separan variando rho).\n")
    L.append("## B. Corridas de referencia (una pasada, orden original)\n")
    L.append("| etapa | rho | clusters | pureza online | esperado | coincide |")
    L.append("|---|---|---|---|---|---|")
    for c in out["corridas"]:
        ok = c["clusters"] == c["esperado_clusters"] and abs(c["pureza_online_pct"] - c["esperado_pureza_pct"]) < 0.01
        L.append(f"| {c['etapa']} | {c['rho']} | {c['clusters']} | {c['pureza_online_pct']}% | "
                 f"{c['esperado_clusters']} / {c['esperado_pureza_pct']}% | {'si' if ok else '**NO**'} |")
    if "fotos" in out:
        r = out["resumen_fotos"]
        L.append("\n## C. Evaluacion fotografica con los modelos entregados (reinferencia: incluye fotos de entrenamiento)\n")
        L.append("| medida | obtenido | esperado | coincide |")
        L.append("|---|---|---|---|")
        for k, lbl in (("tipo_correcto", "Tipo correcto (de 339)"), ("manzanas_derivadas", "Manzanas derivadas a calidad (de 219)"),
                       ("cadena_completa", "Tipo y calidad correctos (de 219)"),
                       ("no_manzanas_derivadas", "Naranjas/bananas derivadas a calidad (de 120)")):
            L.append(f"| {lbl} | {r[k]} | {ESPERADO_FOTOS[k]} | {'si' if r[k] == ESPERADO_FOTOS[k] else '**NO**'} |")
        L.append(f"\nExactitud de tipo {100 * r['exactitud_tipo']:.1f}%, F1 macro {r['f1_macro_tipo']:.3f}; "
                 f"referencia 'siempre carozo': {100 * r['mayoritaria_tipo']['exactitud']:.1f}% / F1 macro "
                 f"{r['mayoritaria_tipo']['f1_macro']:.3f}. Sin resonancia en etapa 1: {r['sin_resonancia_etapa1']}.\n")
        L.append("| estrato | fotos | tipo correcto | derivadas a calidad |")
        L.append("|---|---|---|---|")
        for e, d in r["por_estrato"].items():
            L.append(f"| {e} | {d['n']} | {d['tipo_correcto']} | {d['derivadas']} |")
    h = out["hallazgos"]
    L.append("\n## D. Otros hallazgos comprobados\n")
    for etapa, d in h["mapeo_online_vs_prototipos_finales"].items():
        L.append(f"- **Mapeo online vs prototipos finales ({etapa}, rho={d['rho_modelo_entregado']})**: "
                 f"{d['muestras_sin_resonancia_con_prototipos_finales']}/{d['n_muestras']} muestras de entrenamiento ya no "
                 f"resuenan con los prototipos finales; {d['muestras_cuya_categoria_cambia_al_reevaluar']} cambian de categoria; "
                 f"{d['categorias_cuya_etiqueta_cambia_o_queda_vacia']}/{d['categorias']} categorias cambiarian de etiqueta o quedarian "
                 f"vacias. El modelo entregado {'coincide' if d['modelo_igual_a_reentrenar_su_csv'] else 'NO coincide'} con reentrenar su CSV con su rho.")
    v = h["valor_invalido_pasa_a_metadato"]
    L.append(f"- **Valor invalido pasa a metadato**: caracteristicas detectadas {v['features_detectadas']}, "
             f"metadatos {v['metadatos_detectados']} ({v['nota']}).")
    c = h["corrida_infructuosa_codigo_de_salida"]
    L.append(f"- **Corrida infructuosa**: `{c['comando']}` termino con codigo de salida **{c['codigo']}** "
             f"(salida: {c['stdout']}).")
    L.append("- **rho en rutas de regeneracion**: modelo entregado de morfologia con rho="
             f"{h['rho_en_rutas_de_regeneracion']['modelo_entregado_morfologia']}; lineas que usan otro valor:")
    for l in h["rho_en_rutas_de_regeneracion"]["lineas"]:
        L.append(f"  - `{l}`")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Reproduce la linea de base del primer avance")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sin-imagenes", action="store_true", help="omite la evaluacion fotografica")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "resultados"), help="carpeta de salida")
    a = ap.parse_args()
    if a.worker:
        json.dump(_worker(a.sin_imagenes), sys.stdout)
        return 0
    if not os.path.isdir(BASE):
        print(f"ERROR: falta la carpeta del snapshot '{BASE}'", file=sys.stderr)
        return 3
    cmd = [sys.executable, os.path.abspath(__file__), "--worker"] + (["--sin-imagenes"] if a.sin_imagenes else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print("ERROR en el worker de la linea de base:\n" + p.stderr, file=sys.stderr)
        return 4
    out = json.loads(p.stdout)
    if "fotos" in out:
        out["resumen_fotos"] = _resumen_fotos(out["fotos"])
    os.makedirs(a.salida, exist_ok=True)
    out["entorno"] = {"python": platform.python_version(), "so": platform.platform(), "commit_base": COMMIT_BASE}
    with open(os.path.join(a.salida, "baseline_original.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    _escribir_md(out, os.path.join(a.salida, "baseline_original.md"))
    if "fotos" in out:
        import csv
        cols = list(out["fotos"][0])
        with open(os.path.join(a.salida, "baseline_fotos.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out["fotos"])
    print(open(os.path.join(a.salida, "baseline_original.md"), encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
