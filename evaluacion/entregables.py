"""
evaluacion/entregables.py
Construye los entregables de la version mejorada a partir de una configuracion
elegida: manifiesto y datasets CSV (entrenamiento / validacion / prueba), modelos
ART1 de formato 2 y las tablas comparativas contra el primer avance.

Todo sale de carpetas NUEVAS (datasets_v2/, modelos_v2/, resultados/): los
datasets y modelos del primer avance (datasets/, modelos/) no se tocan.
"""
import csv
import os

import extraccion as X
import validacion as val
from CarGross import cargar_modelo, entrenar_desde_dataset
from evaluacion import banco, cadena, experimentos as E, metricas
from validacion import ALIAS_MANZANA


def _escribir_csv(ruta, columnas, filas):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columnas)
        w.writerows(filas)


def adjuntar_trazabilidad(fotos, ruta_traza):
    """Agrega id_orig_morfologia / id_orig_calidad a cada foto (si existe el CSV de trazabilidad)."""
    if not os.path.isfile(ruta_traza):
        return fotos
    por = {}
    with open(ruta_traza, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            por.setdefault(r["ruta"], {})[r["dataset"]] = r["fruta_id_original"]
    for fo in fotos:
        d = por.get(fo["ruta"], {})
        fo["id_orig_morfologia"], fo["id_orig_calidad"] = d.get("morfologia", ""), d.get("calidad", "")
    return fotos


def escribir_datasets(dir_datasets, fs, por, entr, pres, med):
    """Escribe el manifiesto y los CSV por etapa y particion. Devuelve {(etapa, particion): ruta}."""
    banco.escribir_manifiesto(fs, os.path.join(dir_datasets, "manifiesto_imagenes.csv"))
    rutas = {}
    for etapa, campo, nombre in (("morfologia", "tipo_real", "morfologia"), ("calidad", "calidad_real", "calidad")):
        pre = pres[etapa]
        cols = ["fruta_id"] + pre["cols_features"] + [campo]
        for part, lista in (("entrenamiento", entr), ("validacion", por["validacion"]), ("prueba", por["prueba"])):
            fl = [f for f in lista if etapa == "morfologia" or f["tipo_real"] == ALIAS_MANZANA]
            filas = [[f["foto_id"]] + X.vectorizar(med[f["foto_id"]], pre) + [f[campo]] for f in fl]
            ruta = os.path.join(dir_datasets, f"{nombre}_{part}.csv")
            _escribir_csv(ruta, cols, filas)
            rutas[(etapa, part)] = ruta
        filas = [[c, pre["reglas"][c]["crudo"], pre["reglas"][c]["op"], repr(pre["umbrales"][c])]
                 for c in pre["cols_features"]]
        _escribir_csv(os.path.join(dir_datasets, f"umbrales_{etapa}.csv"),
                      ["variable", "medida_continua", "operador", "umbral"], filas)
    return rutas


def entrenar_y_guardar(var, etapa, ruta_train, rho, pre, ruta_modelo, procedencia, sobrescribir):
    """Entrena desde el CSV (la misma via que el CLI de CarGross.py) y guarda el modelo."""
    ds = val.leer_dataset(ruta_train)
    red, meta = entrenar_desde_dataset(ds, rho, L=2.0, max_categorias=E.TOPE_CATEGORIAS, epocas=var.epocas,
                                       orden="original", politica_nulo=var.politica_nulo,
                                       desempate=var.desempate, modo_mapeo=var.mapeo)
    meta["procedencia"] = procedencia
    pre = dict(pre)
    pre["notas"] = ("'carozo' es el alias historico de manzana. densidad_alta, es_pesada y textura_firme son "
                    "aproximaciones visuales (solidez del contorno, area x solidez, varianza del laplaciano): "
                    "no miden densidad fisica, masa ni firmeza mecanica.")
    red.guardar(ruta_modelo, meta, preprocesamiento=pre, sobrescribir=sobrescribir)
    return cargar_modelo(ruta_modelo)


def vectores(fl, med, pre):
    return {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pre) for f in fl}


def aciertos(regs):
    """(e1_ok, cadena_ok) por foto; cadena_ok: manzana = derivada y calidad correcta; otras = tipo correcto."""
    e1 = [r["morf_etiqueta"] == r["tipo_real"] for r in regs]
    ca = []
    for r in regs:
        if r["tipo_real"] == ALIAS_MANZANA:
            ca.append(bool(r["derivada"] and r.get("cal_etiqueta") == r["calidad_real"]))
        else:
            ca.append(r["morf_etiqueta"] == r["tipo_real"])
    return e1, ca


def ic(regs, indicador):
    ok = aciertos(regs)[indicador]
    lo, hi = metricas.ic_bootstrap_por_grupos(ok, [r["grupo"] for r in regs])
    return lo, hi


def pct(x):
    return "-" if x is None or x != x else f"{100 * x:.1f}%"


def tabla_resumen(filas_sistemas, titulo_cols):
    """Tabla markdown. filas_sistemas: [(nombre, resumen_dict, ic_cadena|None)]."""
    cab = "| sistema | " + " | ".join(t for t, _ in titulo_cols) + " |"
    sep = "|---|" + "---|" * len(titulo_cols)
    L = [cab, sep]
    for nombre, r, extra in filas_sistemas:
        celdas = []
        for _, f in titulo_cols:
            celdas.append(f(r, extra))
        L.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    return "\n".join(L)


COLUMNAS_PRINCIPALES = [
    ("fotos", lambda r, e: str(r["n"])),
    ("tipo correcto", lambda r, e: f"{r['e1_tipo_correcto']}/{r['n']} ({pct(r['e1_exactitud'])})"),
    ("F1 macro tipo", lambda r, e: f"{r['e1_f1_macro']:.3f}"),
    ("sin identificar (etapa 1)", lambda r, e: str(r["e1_novedades"] + r["e1_sin_etiqueta"])),
    ("manzanas derivadas a calidad", lambda r, e: f"{r['manzanas_derivadas']}/{r['n_manzanas']}"),
    ("naranjas/bananas derivadas", lambda r, e: f"{r['no_manzanas_derivadas']}/{r['n_no_manzanas']}"),
    ("calidad aislada: exactitud / F1", lambda r, e: f"{pct(r.get('e2_aislada_exactitud'))} / {r.get('e2_aislada_f1_macro', float('nan')):.3f}"),
    ("cadena manzanas (tipo+calidad)", lambda r, e: f"{r['cadena_manzanas_correctas']}/{r['n_manzanas']} ({pct(r['cadena_manzanas_exactitud'])})"),
    ("cadena total", lambda r, e: f"{r['cadena_total_correcta']}/{r['n']} ({pct(r['cadena_total_exactitud'])})" +
     (f" IC95 [{pct(e[0])}, {pct(e[1])}]" if e else "")),
    ("referencia mayoritaria (tipo)", lambda r, e: f"{pct(r['linea_base_tipo']['exactitud'])} / F1 {r['linea_base_tipo']['f1_macro']:.3f}"),
]


def matriz_md(conf, clases, titulo):
    cols = clases + [metricas.SIN_ETIQUETA]
    L = [f"**{titulo}**\n", "| real \\ predicha | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in clases:
        L.append(f"| {c} | " + " | ".join(str(conf[c].get(k, 0)) for k in cols) + " |")
    return "\n".join(L)


def evaluar_original_entregado(fotos_part, med_v1, dir_modelos_original):
    """Modelos entregados (formato 1) con el preprocesamiento del primer avance."""
    mm = cargar_modelo(os.path.join(dir_modelos_original, "art1_morfologia.json"))
    mc = cargar_modelo(os.path.join(dir_modelos_original, "art1_calidad.json"))
    pm, pc = X.preprocesamiento_legado("morfologia"), X.preprocesamiento_legado("calidad")
    vm, vc = vectores(fotos_part, med_v1, pm), vectores(fotos_part, med_v1, pc)
    regs = cadena.evaluar_cadena(fotos_part, vm, vc, mm, mc)
    return regs
