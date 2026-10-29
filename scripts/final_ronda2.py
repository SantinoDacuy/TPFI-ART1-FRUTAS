"""
scripts/final_ronda2.py
Genera los entregables de la segunda ronda a partir de la configuracion principal elegida en
scripts/busqueda.py y de los rho de scripts/comparar_ronda2.py (resultados/ronda2/comparacion.json):

  datasets_v2/manifiesto_imagenes.csv                  manifiesto con el bloque (desarrollo / informe)
  datasets_v2/<variante>/{morfologia,calidad}_{entrenamiento,informe}.csv      entrada de la red
  datasets_v2/<variante>/preprocesamiento_{morfologia,calidad}.csv             el preprocesamiento exacto (CSV)
  modelos_v2/<variante>/art1_{morfologia,calidad}.json                          modelos formato 2 de fotografias
  config/parametros_etapas.csv                         parametros de entrenamiento de la variante principal

Variantes generadas: 'principal' (compatible con el primer avance), 'alternativa_orden_canonico' (compatible, orden de
presentacion determinista) y 'experimental_medianas'. Los archivos de la
primera ronda se CONSERVAN (se mueven a datasets_v2/ronda1_medianas/ y modelos_v2/ronda1_medianas/ solo si no
existian ya ahi; nunca se sobrescriben sin --sobrescribir).

El modelo se entrena DESDE EL CSV escrito (la misma via que `CarGross.py --csv ...`) y se comprueba que coincide con
el entrenado en memoria y que, recargado desde JSON, predice igual sobre el informe.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import replace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

import busqueda as B
import comparar_ronda2 as C
import configuracion as cfg
import extraccion as X
import validacion as val
from CarGross import cargar_modelo, entrenar_desde_dataset
from evaluacion import banco, cadena, experimentos as E, metricas
from evaluacion import entregables as EN
from validacion import COD_ENTRADA, COD_OK, COD_USO

ETAPAS = (("morfologia", "tipo_real"), ("calidad", "calidad_real"))


def cfg_hash(ruta):
    from CarGross import hash_archivo
    return hash_archivo(ruta)


def _mover_ronda1(dir_datasets, dir_modelos):
    """Conserva los archivos de la primera ronda (sin borrar nada): se MUEVEN a carpetas 'ronda1_medianas'."""
    d1 = os.path.join(dir_datasets, "ronda1_medianas")
    m1 = os.path.join(dir_modelos, "ronda1_medianas")
    movidos = []
    for carpeta, destino, patron in ((dir_datasets, d1, lambda n: n.endswith(".csv")
                                      and n not in ("trazabilidad_ids_originales.csv", "manifiesto_imagenes.csv")),
                                     (dir_modelos, m1, lambda n: n.endswith(".json"))):
        if not os.path.isdir(carpeta):
            continue
        for n in sorted(os.listdir(carpeta)):
            ruta = os.path.join(carpeta, n)
            if os.path.isfile(ruta) and patron(n):
                os.makedirs(destino, exist_ok=True)
                if os.path.exists(os.path.join(destino, n)):
                    raise FileExistsError(f"ya existe {os.path.join(destino, n)}: no se sobrescribe la primera ronda")
                shutil.copy2(ruta, os.path.join(destino, n))      # se COPIA primero: el original se retira solo despues
                movidos.append(ruta)
    return movidos


def generar(nombre, var, rho, fotos, med, fs, dir_ds, dir_mod, procedencia, sobrescribir):
    fin = E.ajustar_desarrollo(var, fotos, med[E.clave_mediciones(var)], B.SEMILLA_INFORME, rho[0], rho[1])
    if "prueba" not in fin:
        raise RuntimeError(f"corrida infructuosa para '{nombre}'")
    pres = fin["_pre"]
    por = {"entrenamiento": [f for f in fs if f["bloque"] == "desarrollo"],
           "informe": [f for f in fs if f["bloque"] == "informe"]}
    carpeta_ds = os.path.join(dir_ds, nombre)
    carpeta_m = os.path.join(dir_mod, nombre)
    modelos = {}
    for etapa, campo in ETAPAS:
        pre = pres[etapa]
        orden = fin["_orden_entrenamiento"][etapa]                  # ids en el orden de presentacion usado
        por_id = {f["foto_id"]: f for f in fs}
        filas_tr = [por_id[i] for i in orden]
        filas_inf = [f for f in por["informe"] if etapa == "morfologia" or f["tipo_real"] == "carozo"]
        for part, fl in (("entrenamiento", filas_tr), ("informe", filas_inf)):
            ruta = os.path.join(carpeta_ds, f"{etapa}_{part}.csv")
            if os.path.exists(ruta) and not sobrescribir:
                raise FileExistsError(f"'{ruta}' ya existe; use --sobrescribir")
            os.makedirs(carpeta_ds, exist_ok=True)
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["fruta_id"] + pre["cols_features"] + [campo])
                for fo in fl:
                    w.writerow([fo["foto_id"]] + X.vectorizar(med[E.clave_mediciones(var)][fo["foto_id"]], pre) + [fo[campo]])
        ruta_pre = os.path.join(carpeta_ds, f"preprocesamiento_{etapa}.csv")
        if os.path.exists(ruta_pre) and not sobrescribir:
            raise FileExistsError(f"'{ruta_pre}' ya existe; use --sobrescribir")
        X.escribir_preprocesamiento_csv(pre, ruta_pre)
        # entrenamiento DESDE EL CSV (misma via que el CLI) con el preprocesamiento leido del CSV documentado
        ds = val.leer_dataset(os.path.join(carpeta_ds, f"{etapa}_entrenamiento.csv"))
        rho_e = rho[0] if etapa == "morfologia" else rho[1]
        red, meta = entrenar_desde_dataset(ds, rho_e, L=2.0, max_categorias=E.TOPE_CATEGORIAS, epocas=var.epocas,
                                           orden="original", politica_nulo=var.politica_nulo,
                                           desempate=var.desempate, modo_mapeo=var.mapeo,
                                           soporte_min=var.soporte_min)
        mem = fin["_modelos"][etapa]
        assert red.top_down == mem.red.top_down, f"{nombre}/{etapa}: el modelo desde el CSV difiere del evaluado"
        assert json.loads(json.dumps(meta["mapeo_clusters"])) == json.loads(json.dumps(mem.meta["mapeo_clusters"]))
        pre_csv = X.leer_preprocesamiento_csv(ruta_pre, ds.cols_features)
        meta["procedencia"] = {**procedencia, "variante": nombre, "rho": rho_e,
                               "orden_presentacion": f"{var.orden} (fijado en el orden de filas del CSV de entrenamiento)",
                               "umbrales_calculados_con": "solo las fotos de desarrollo (sin el bloque de informe)"}
        meta["preprocesamiento_origen"] = os.path.relpath(ruta_pre, RAIZ).replace(os.sep, "/")
        meta["comprometido_con_el_primer_avance"] = nombre == "principal"
        destino = os.path.join(carpeta_m, f"art1_{etapa}.json")
        if os.path.exists(destino) and not sobrescribir:
            raise FileExistsError(f"'{destino}' ya existe; use --sobrescribir")
        red.guardar(destino, meta, preprocesamiento=pre_csv, sobrescribir=sobrescribir)
        modelos[etapa] = cargar_modelo(destino)
    # evaluacion con los modelos recargados == evaluacion en memoria
    inf = por["informe"]
    vm = {f["foto_id"]: X.vectorizar(med[E.clave_mediciones(var)][f["foto_id"]], modelos["morfologia"].preprocesamiento) for f in inf}
    vc = {f["foto_id"]: X.vectorizar(med[E.clave_mediciones(var)][f["foto_id"]], modelos["calidad"].preprocesamiento) for f in inf}
    regs = cadena.evaluar_cadena(inf, vm, vc, modelos["morfologia"], modelos["calidad"], var.derivacion_min)
    previo = {r["foto_id"]: (r["morf_etiqueta"], r.get("cal_etiqueta")) for r in fin["prueba"]["_registros"]}
    assert all(previo[r["foto_id"]] == (r["morf_etiqueta"], r.get("cal_etiqueta")) for r in regs), \
        "guardar/cargar o el entrenamiento desde CSV alteraron las predicciones"
    return fin, modelos


def main(argv=None):
    ap = argparse.ArgumentParser(description="Entregables de la segunda ronda (datasets, modelos, config)")
    ap.add_argument("--principal", default=os.path.join(B.DIR_SALIDA, "config_principal.json"))
    ap.add_argument("--comparacion", default=os.path.join(B.DIR_SALIDA, "comparacion.json"))
    ap.add_argument("--mediciones", default=os.path.join(cfg.DIR_RESULTADOS, "mediciones"))
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    ap.add_argument("--datasets-v2", default=cfg.DIR_DATASETS_V2)
    ap.add_argument("--modelos-v2", default=cfg.DIR_MODELOS_V2)
    ap.add_argument("--sobrescribir", action="store_true")
    a = ap.parse_args(argv)
    for r in (a.principal, a.comparacion):
        if not os.path.isfile(r):
            print(f"ERROR: falta '{r}'. Ejecute antes scripts/busqueda.py buscar y scripts/comparar_ronda2.py", file=sys.stderr)
            return COD_ENTRADA
    principal = C.cargar_principal(a.principal)
    with open(a.comparacion, encoding="utf-8") as f:
        comp = json.load(f)
    vs = C.variantes(principal)
    fotos = banco.inventariar(a.imagenes)
    EN.adjuntar_trazabilidad(fotos, os.path.join(a.datasets_v2, "trazabilidad_ids_originales.csv"))
    med = E.cargar_todas_las_mediciones(a.mediciones)
    fs = [dict(f) for f in fotos]
    banco.reservar_informe(fs, B.SEMILLA_INFORME)
    try:
        movidos = _mover_ronda1(a.datasets_v2, a.modelos_v2)
        manif1 = os.path.join(a.datasets_v2, "manifiesto_imagenes.csv")
        if os.path.isfile(manif1) and "bloque" not in open(manif1, encoding="utf-8").readline():
            os.makedirs(os.path.join(a.datasets_v2, "ronda1_medianas"), exist_ok=True)
            shutil.copy2(manif1, os.path.join(a.datasets_v2, "ronda1_medianas", "manifiesto_imagenes_ronda1.csv"))
        for ruta in movidos:      # mover = copia verificada + retiro del nivel superior (no se pierde informacion)
            copia = os.path.join(os.path.dirname(ruta), "ronda1_medianas", os.path.basename(ruta))
            if cfg_hash(ruta) != cfg_hash(copia):
                raise RuntimeError(f"la copia de respaldo de {ruta} no coincide; no se retira el original")
            os.remove(ruta)
        banco.escribir_manifiesto_ronda2(fs, manif1)
        procedencia = {"semilla_informe": B.SEMILLA_INFORME, "manifiesto": "datasets_v2/manifiesto_imagenes.csv",
                       "script": "scripts/final_ronda2.py", "banco": "339 fotos ya inspeccionadas (no es un conjunto externo)"}
        resultados = {}
        for nombre_dir, clave in (("principal", "principal"), ("alternativa_orden_canonico", "alt_orden_canonico"),
                                    ("experimental_medianas", "exp_medianas"),
                                    ("experimental_segmentacion_v2", "exp_segmentacion_v2"),
                                    ("experimental_normalizacion", "exp_normalizacion")):
            var = vs[clave][0]
            rho = tuple(comp[clave]["rho"])
            fin, modelos = generar(nombre_dir, var, rho, fotos, med, fs, a.datasets_v2, a.modelos_v2, procedencia,
                                   a.sobrescribir)
            resultados[nombre_dir] = (var, rho)
            print(f"{nombre_dir}: rho {rho}, categorias {len(modelos['morfologia'].red.top_down)} / "
                  f"{len(modelos['calidad'].red.top_down)}")
        var, rho = resultados["principal"]
        with open(os.path.join(cfg.DIR_RAIZ, "config", "parametros_etapas.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["etapa", "rho", "L", "max_categorias", "epocas", "politica_nulo", "desempate", "fuente"])
            for etapa, r in zip(("morfologia", "calidad"), rho):
                w.writerow([etapa, r, 2.0, E.TOPE_CATEGORIAS, var.epocas, var.politica_nulo, var.desempate,
                            "variante principal compatible con el primer avance (scripts/final_ronda2.py); rho: mediana de los "
                            "elegidos en validacion interna, <= 14 categorias"])
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return COD_USO
    return COD_OK


if __name__ == "__main__":
    sys.exit(main())
