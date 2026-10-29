"""
evaluacion/experimentos.py
Experimentos reproducibles sobre el banco de 339 fotos, a partir de las
mediciones continuas ya calculadas (resultados/mediciones/*.csv): no requiere
las fotos.

Protocolo (por cada semilla de particion):
  1. Las fotos se reparten por GRUPO (rafaga/misma toma probable) en
     entrenamiento / validacion / prueba, estratificado por origen.
     - Ronda 1: reparto 3-vias de todas las fotos (banco.particionar).
     - Ronda 2: se reserva un bloque de 'informe' (banco.reservar_informe) que hace de 'prueba' en todas las
       semillas; entrenamiento/validacion salen solo de 'desarrollo' (banco.particionar_desarrollo).
  2. Los umbrales de binarizacion que dependen de los datos se calculan SOLO con
     el entrenamiento.
  3. rho se elige sobre VALIDACION (barrido) por F1 macro de cada etapa (o por la cadena, segun la variante);
     ante empate, menos categorias y luego menor rho.
  4. Con esos parametros se entrena sobre entrenamiento y se mide en tres
     lugares: reinferencia sobre entrenamiento, validacion y prueba.
La decision entre variantes se toma con validacion; prueba se informa aparte.
"""
import csv
import random
from collections import Counter
from dataclasses import dataclass

import extraccion as X
from CarGross import (ProliferacionError, RedART1, Modelo, construir_mapeo, entrenar, mapeo_online,
                      pureza)
from evaluacion import banco, cadena, metricas
from validacion import ALIAS_MANZANA

GRILLA_RHO = tuple(round(0.30 + 0.05 * i, 2) for i in range(15))   # 0.30 ... 1.00
TOPE_CATEGORIAS = 50
ESTRATOS_BUENOS = ("manzana_good",)   # composicion original de la etapa 1: solo manzanas "good"
ORDENES = ("aleatorio", "densidad_desc", "densidad_asc", "canonico_asc")


@dataclass(frozen=True)
class Variante:
    nombre: str
    segmentador: str = "v2"
    legado: bool = False             # True: reglas/umbrales exactos del primer avance
    tallo: str = "protuberancia"
    tono: str = "circular"
    extras_morf: tuple = ()
    extras_cal: tuple = ()
    umbral_morf: str = "fijo"
    umbral_cal: str = "mediana"
    cobertura: str = "todas"         # 'buenas' (original) | 'todas' las manzanas en etapa 1
    mapeo: str = "prototipos_finales"
    desempate: str = "indeterminada"
    rho_morf: float = None           # None -> elegir en validacion
    rho_cal: float = None
    epocas: int = 1
    politica_nulo: str = "categoria_propia"
    # --- ronda 2 (los valores por defecto reproducen la ronda 1) ---
    color: str = "mascara"           # medicion de los descriptores cromaticos adicionales
    soporte_min: int = 1             # muestras minimas para que una categoria lleve etiqueta
    derivacion_min: float = 0.0      # coincidencia minima de la etapa 1 para derivar a calidad (0 = solo resonar)
    criterio_rho: str = "f1_etapa"   # 'f1_etapa' | 'cadena' (rho de la etapa 1 por la cadena en validacion)
    tope_k: int = None               # maximo de categorias admisible al elegir rho (criterio de exito del primer avance: < 15)
    orden: str = "aleatorio"         # orden de presentacion durante el entrenamiento
    comite: int = 1                  # EXPERIMENTAL: n de redes ART1 (ordenes distintos) que votan la etiqueta
    normalizacion: str = "ninguna"   # paso previo de iluminacion (preprocesador.NORMALIZACIONES)


# Primer avance reentrenado sobre la particion limpia (mismos pasos y parametros que el entregado)
ORIGINAL = Variante("original", segmentador="v1", legado=True, cobertura="buenas", mapeo="online",
                    rho_morf=0.85, rho_cal=0.70, umbral_cal="fijo")


def clave_mediciones(var):
    """Archivo de mediciones que corresponde a la variante: (segmentador, normalizacion)."""
    return (var.segmentador, var.normalizacion)


def cargar_todas_las_mediciones(carpeta):
    """{(segmentador, normalizacion): {foto_id: crudos}} con todos los mediciones_<seg>[_<norm>].csv disponibles."""
    import os
    out = {}
    for n in sorted(os.listdir(carpeta)):
        if n.startswith("mediciones_") and n.endswith(".csv"):
            partes = n[len("mediciones_"):-4].split("_", 1)
            norm = partes[1] if len(partes) > 1 else "ninguna"
            out[(partes[0], norm)] = leer_mediciones(os.path.join(carpeta, n))
    return out


def leer_mediciones(ruta_csv):
    """{foto_id: {medida: float}} (las columnas de texto del segmentador se omiten)."""
    out = {}
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            out[fila["foto_id"]] = {k: float(v) for k, v in fila.items()
                                    if k not in ("foto_id", "seg_candidato", "seg_advertencia") and v != ""}
    return out


def leer_info_segmentacion(ruta_csv):
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        return {r["foto_id"]: {"candidato": r["seg_candidato"], "puntaje": r["seg_puntaje"],
                               "advertencia": r["seg_advertencia"]} for r in csv.DictReader(f)}


def _pre(var, etapa, crudos_train):
    if var.legado:
        return X.preprocesamiento_legado(etapa)
    estrategia = var.umbral_morf if etapa == "morfologia" else var.umbral_cal
    extras = var.extras_morf if etapa == "morfologia" else var.extras_cal
    um = X.umbrales_desde_entrenamiento(etapa, crudos_train, estrategia, var.tallo, var.tono, extras, var.color)
    return X.configuracion(etapa, um, var.segmentador, var.tallo, var.tono, extras,
                           origen_umbrales=f"estrategia '{estrategia}' sobre el entrenamiento", color=var.color,
                           normalizacion=var.normalizacion)


class _RedComite:
    """Interfaz minima de RedART1 (evaluar / huella) para un COMITE de redes ART1 entrenadas con ordenes distintos.
    EXPERIMENTAL: cada red conserva su propio aprendizaje y etiquetado; la etiqueta final es la mayoritaria entre
    las redes que resuenan con una categoria etiquetada (empate o ninguna = sin decision). Costo: K redes."""

    def __init__(self, miembros):
        self.miembros = miembros                      # lista de (RedART1, mapeo_clusters)
        self.rho = miembros[0][0].rho

    def evaluar(self, x):
        votos, ratios = Counter(), {}
        mejor = 0.0
        for red, mapeo in self.miembros:
            ev = red.evaluar(x)
            mejor = max(mejor, ev["ratio"])
            if ev["resuena"]:
                et = mapeo[str(ev["ganador"])]["etiqueta"]
                if et:
                    votos[et] += 1
                    ratios.setdefault(et, []).append(ev["ratio"])
        if votos:
            tope = max(votos.values())
            lideres = sorted(e for e, c in votos.items() if c == tope)
            if len(lideres) == 1:
                e = lideres[0]
                return {"ganador": e, "ratio": sum(ratios[e]) / len(ratios[e]), "resuena": True}
        return {"ganador": None, "ratio": mejor, "resuena": False, "motivo": "comite"}

    def huella(self):
        return "|".join(r.huella() for r, _ in self.miembros)


class ModeloComite(Modelo):
    def categoria(self, cid):
        return {"etiqueta": cid, "estado": "comite", "n": 0, "pureza": None}


def _entrenar_comite(var, vectores, etiquetas, rho, pre, semilla_orden):
    miembros, ks, info0 = [], [], None
    for k in range(var.comite):
        idx = list(range(len(vectores)))
        random.Random(semilla_orden * 1000 + k).shuffle(idx)
        v = [vectores[i] for i in idx]
        y = [etiquetas[i] for i in idx]
        red = RedART1(len(pre["cols_features"]), rho, 2.0, TOPE_CATEGORIAS, var.politica_nulo)
        try:
            info = entrenar(red, v, epocas=var.epocas, orden="original")
        except ProliferacionError as e:
            return None, str(e)
        mapeo, _ = construir_mapeo(red, v, y, var.desempate, soporte_min=var.soporte_min)
        miembros.append((red, mapeo))
        ks.append(len(red.top_down))
        info0 = info0 or {"estable": info["estable"]}
    modelo = ModeloComite(_RedComite(miembros), {"cols_features": list(pre["cols_features"]),
                                                   "mapeo_clusters": miembros[0][1]}, pre, 2, [], None)
    return modelo, {"n_categorias": max(ks), "pureza_online": float("nan"), "pureza_final": float("nan"),
                    "cobertura_final": float("nan"), "estable": info0["estable"], "k_miembros": ks}


def entrenar_etapa(var, etapa, vectores, etiquetas, rho, pre, semilla_orden=0):
    """Entrena una etapa; devuelve (Modelo, info) o (None, motivo) si es infructuosa."""
    if var.comite > 1:
        return _entrenar_comite(var, vectores, etiquetas, rho, pre, semilla_orden)
    red = RedART1(len(pre["cols_features"]), rho, 2.0, TOPE_CATEGORIAS, var.politica_nulo)
    try:
        info = entrenar(red, vectores, epocas=var.epocas, orden="original")
    except ProliferacionError as e:
        return None, str(e)
    if var.mapeo == "online":
        mapeo = mapeo_online(info["asignaciones_online"], etiquetas)
    else:
        mapeo, _ = construir_mapeo(red, vectores, etiquetas, var.desempate, soporte_min=var.soporte_min)
    finales = [r["ganador"] if r["resuena"] else None for r in map(red.evaluar, vectores)]
    p_on, _ = pureza(info["asignaciones_online"], etiquetas)
    p_fin, cob = pureza(finales, etiquetas)
    modelo = Modelo(red, {"cols_features": list(pre["cols_features"]), "mapeo_clusters": mapeo},
                    pre, 2, [], None)
    return modelo, {"n_categorias": len(red.top_down), "pureza_online": p_on, "pureza_final": p_fin,
                    "cobertura_final": cob, "estable": info["estable"]}


def _f1_etapa(modelo, vecs, fotos, etapa):
    """F1 macro de una etapa sobre 'fotos' (para elegir rho)."""
    clases = cadena.TIPOS if etapa == "morfologia" else cadena.CALIDADES
    campo = "tipo_real" if etapa == "morfologia" else "calidad_real"
    if etapa == "calidad":
        fotos = [f for f in fotos if f["tipo_real"] == ALIAS_MANZANA]
    verdad, pred = [], []
    for f in fotos:
        ev = modelo.red.evaluar(vecs[f["foto_id"]])
        pred.append(modelo.categoria(ev["ganador"])["etiqueta"] if ev["resuena"] else None)
        verdad.append(f[campo])
    return metricas.f1_macro(verdad, pred, clases)


def elegir_rho(var, etapa, pre, entr_vecs, entr_et, val_vecs, val_fotos, grilla=GRILLA_RHO, puntaje=None):
    """Barrido de rho sobre VALIDACION. Devuelve (rho, tabla).
    puntaje(modelo) -> numero (mayor = mejor) reemplaza al F1 de la etapa (p. ej. la cadena en validacion).
    Con var.tope_k solo son admisibles los rho cuyo entrenamiento forma <= tope_k categorias (criterio de exito
    del primer avance: menos de 15 clusters para 3 clases); si ninguno lo cumple se toma el de menos categorias."""
    tabla = []
    for rho in grilla:
        m, info = entrenar_etapa(var, etapa, entr_vecs, entr_et, rho, pre)
        if m is None:
            tabla.append({"rho": rho, "ok": False, "f1_val": float("nan"), "k": None})
            continue
        sc = puntaje(m) if puntaje else _f1_etapa(m, val_vecs, val_fotos, etapa)
        tabla.append({"rho": rho, "ok": True, "f1_val": sc, "k": info["n_categorias"]})
    validas = [t for t in tabla if t["ok"]]
    if not validas:
        raise RuntimeError(f"ningun rho de la grilla produjo una corrida exitosa en '{etapa}'")
    if var.tope_k is not None:
        admisibles = [t for t in validas if t["k"] <= var.tope_k]
        validas = admisibles or [min(validas, key=lambda t: (t["k"], t["rho"]))]
    mejor = min(validas, key=lambda t: (-t["f1_val"], t["k"], t["rho"]))
    return mejor["rho"], tabla


def particion_con_semilla(fotos, semilla):
    fs = [dict(f) for f in fotos]
    banco.particionar(fs, semilla)
    return fs


def particion_informe(fotos, semilla, semilla_informe, fraccion=0.3):
    """Ronda 2: bloque de informe reservado (fijo, semilla_informe) + entrenamiento/validacion por semilla."""
    fs = [dict(f) for f in fotos]
    banco.reservar_informe(fs, semilla_informe, fraccion)
    banco.particionar_desarrollo(fs, semilla)
    return fs


def ordenar_entrenamiento(entrenamiento, semilla):
    """Orden de presentacion (aleatorio y fijo por semilla) de las fotos de entrenamiento."""
    orden = list(range(len(entrenamiento)))
    random.Random(semilla).shuffle(orden)
    return [entrenamiento[i] for i in orden]


def _reordenar(vectores, etiquetas, orden, ids):
    """Reordena (sin usar etiquetas) segun 'orden'; el orden aleatorio de partida actua de desempate.
    Devuelve (vectores, etiquetas, ids) en el orden de presentacion."""
    if orden not in ORDENES:
        raise ValueError(f"orden desconocido: {orden!r} (opciones: {ORDENES})")
    if orden == "aleatorio":
        return vectores, etiquetas, ids
    if orden == "canonico_asc":   # determinista: no depende de ningun orden aleatorio, solo del CONJUNTO de patrones
        idx = sorted(range(len(vectores)), key=lambda i: (sum(vectores[i]), tuple(vectores[i])))
        return [vectores[i] for i in idx], [etiquetas[i] for i in idx], [ids[i] for i in idx]
    sgn = -1 if orden == "densidad_desc" else 1
    idx = sorted(range(len(vectores)), key=lambda i: sgn * sum(vectores[i]))      # estable
    return [vectores[i] for i in idx], [etiquetas[i] for i in idx], [ids[i] for i in idx]


def _tamanos(modelo):
    if isinstance(modelo, ModeloComite):    # comite: se informa la red de referencia (la primera)
        n = [i["n"] for i in modelo.meta["mapeo_clusters"].values()]
        return {"n_categorias": max(len(r.top_down) for r, _ in modelo.red.miembros), "tamanos": sorted(n, reverse=True),
                "unitarias": sum(1 for x in n if x == 1), "sin_etiqueta": 0, "comite": len(modelo.red.miembros)}
    n = [i["n"] for i in modelo.meta["mapeo_clusters"].values()]
    sin_et = sum(1 for i in modelo.meta["mapeo_clusters"].values() if i.get("etiqueta") is None)
    return {"n_categorias": len(n), "tamanos": sorted(n, reverse=True),
            "unitarias": sum(1 for x in n if x == 1), "sin_etiqueta": sin_et}


def correr_split(var, fotos, med, semilla, grilla=GRILLA_RHO, fotos_particionadas=None,
                 orden_semilla=None, semilla_informe=None):
    """Ejecuta una variante en una particion. 'med': {foto_id: crudos} del
    segmentador de la variante. Devuelve dict con parametros elegidos y metricas
    de las tres evaluaciones. Con semilla_informe usa el protocolo de la ronda 2."""
    if fotos_particionadas is not None:
        fs = fotos_particionadas
    elif semilla_informe is not None:
        fs = particion_informe(fotos, semilla, semilla_informe)
    else:
        fs = particion_con_semilla(fotos, semilla)
    por = {p: [f for f in fs if f["particion"] == p] for p in banco.PARTICIONES}
    # orden aleatorio fijo por semilla (orden_semilla permite variar SOLO el orden, p. ej. en
    # el analisis de sensibilidad, manteniendo la misma particion)
    entr = ordenar_entrenamiento(por["entrenamiento"], 10_000 + semilla if orden_semilla is None else orden_semilla)

    entr1 = [f for f in entr if var.cobertura == "todas" or f["tipo_real"] != ALIAS_MANZANA
             or f["estrato"] in ESTRATOS_BUENOS]
    entr2 = [f for f in entr if f["tipo_real"] == ALIAS_MANZANA]
    res = {"semilla": semilla, "variante": var.nombre,
           "n": {p: len(v) for p, v in por.items()}}

    modelos, pres, prep, orden_ids = {}, {}, {}, {}
    # la etapa 2 se entrena primero: la cadena en validacion (criterio de rho de la etapa 1) la necesita
    for etapa, grupo, campo in (("calidad", entr2, "calidad_real"), ("morfologia", entr1, "tipo_real")):
        pre = _pre(var, etapa, [med[f["foto_id"]] for f in grupo])
        vec = lambda fl: {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pre) for f in fl}
        entr_v, entr_e, ids_ord = _reordenar([vec([f])[f["foto_id"]] for f in grupo], [f[campo] for f in grupo],
                                             var.orden, [f["foto_id"] for f in grupo])
        orden_ids.setdefault("ids", {})[etapa] = ids_ord
        rho = var.rho_morf if etapa == "morfologia" else var.rho_cal
        tabla, puntaje = None, None
        if rho is None:
            vval = vec(por["validacion"])
            if etapa == "morfologia" and var.criterio_rho == "cadena":
                vc = {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pres["calidad"]) for f in por["validacion"]}

                def puntaje(m, _vm=vval, _vc=vc):
                    regs = cadena.evaluar_cadena(por["validacion"], _vm, _vc, m, modelos["calidad"], var.derivacion_min)
                    return cadena.resumir_cadena(regs)["cadena_total_exactitud"]
            rho, tabla = elegir_rho(var, etapa, pre, entr_v, entr_e, vval, por["validacion"], grilla, puntaje)
        m, info = entrenar_etapa(var, etapa, entr_v, entr_e, rho, pre)
        if m is None:
            res[f"fallo_{etapa}"] = info
            return res
        modelos[etapa], pres[etapa] = m, pre
        res[f"rho_{etapa}"] = rho
        res[f"entrenamiento_{etapa}"] = {**info, **_tamanos(m)}
        res[f"n_entrenamiento_{etapa}"] = len(grupo)
        if tabla:
            res[f"barrido_{etapa}"] = tabla
    may_tipo = metricas.clase_mayoritaria([f["tipo_real"] for f in entr1])
    may_cal = metricas.clase_mayoritaria([f["calidad_real"] for f in entr2])
    res["mayoritarias"] = {"tipo": may_tipo, "calidad": may_cal}
    for nombre, fl in (("entrenamiento", por["entrenamiento"]), ("validacion", por["validacion"]),
                       ("prueba", por["prueba"])):
        vm = {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pres["morfologia"]) for f in fl}
        vc = {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pres["calidad"]) for f in fl}
        regs = cadena.evaluar_cadena(fl, vm, vc, modelos["morfologia"], modelos["calidad"], var.derivacion_min)
        res[nombre] = cadena.resumir_cadena(regs, may_tipo, may_cal)
        res[nombre]["_registros"] = regs
    res["_modelos"] = modelos
    res["_pre"] = pres
    res["_orden_entrenamiento"] = orden_ids.get("ids", {})
    return res


def ajustar_desarrollo(var, fotos, med, semilla_informe, rho_morf, rho_cal, semilla_orden=0):
    """Ajuste FINAL: entrena con TODAS las fotos de desarrollo (entrenamiento + validacion de las particiones
    internas) con rho dados y evalua en el bloque de informe. Devuelve el mismo dict que correr_split."""
    fs = [dict(f) for f in fotos]
    banco.reservar_informe(fs, semilla_informe)
    for f in fs:
        f["particion"] = "entrenamiento" if f["bloque"] == "desarrollo" else "prueba"
    v = Variante(**{**var.__dict__, "rho_morf": rho_morf, "rho_cal": rho_cal})
    return correr_split(v, fotos, med, semilla_orden, fotos_particionadas=fs)


# ---------------------------------------------------------------------------
# Agregacion entre semillas
# ---------------------------------------------------------------------------
METRICAS_CLAVE = ("e1_exactitud", "e1_f1_macro", "e1_novedades", "manzanas_derivadas",
                  "no_manzanas_derivadas", "cadena_manzanas_exactitud", "cadena_total_exactitud",
                  "e2_aislada_exactitud", "e2_aislada_f1_macro", "e2_aislada_novedades")


def agregar(resultados, particion):
    """{metrica: (media, desvio)} entre semillas para una particion; mas rho y k elegidos."""
    ok = [r for r in resultados if particion in r]
    out = {"n_semillas": len(ok), "fallos": len(resultados) - len(ok)}
    for k in METRICAS_CLAVE:
        out[k] = metricas.media_y_desvio([r[particion].get(k, float("nan")) for r in ok])
    for etapa in ("morfologia", "calidad"):
        out[f"rho_{etapa}"] = metricas.media_y_desvio([r[f"rho_{etapa}"] for r in ok])
        out[f"k_{etapa}"] = metricas.media_y_desvio([r[f"entrenamiento_{etapa}"]["n_categorias"] for r in ok])
        out[f"unitarias_{etapa}"] = metricas.media_y_desvio(
            [r[f"entrenamiento_{etapa}"].get("unitarias", float("nan")) for r in ok])
        out[f"pureza_online_{etapa}"] = metricas.media_y_desvio([r[f"entrenamiento_{etapa}"]["pureza_online"] for r in ok])
        out[f"pureza_final_{etapa}"] = metricas.media_y_desvio([r[f"entrenamiento_{etapa}"]["pureza_final"] for r in ok])
    out["linea_base_tipo"] = metricas.media_y_desvio([r[particion]["linea_base_tipo"]["exactitud"] for r in ok])
    out["linea_base_cadena_manzanas"] = metricas.media_y_desvio(
        [r[particion].get("linea_base_cadena_manzanas", float("nan")) for r in ok])
    return out


def correr_semillas(var, fotos, med, semillas, grilla=GRILLA_RHO):
    return [correr_split(var, fotos, med, s, grilla) for s in semillas]


def diferencia_pareada(res_a, res_b, particion, metrica="cadena_total_exactitud"):
    """Media y error estandar de (b - a) entre semillas, emparejando por semilla."""
    d = []
    for ra, rb in zip(res_a, res_b):
        if particion in ra and particion in rb:
            d.append(rb[particion][metrica] - ra[particion][metrica])
    if not d:
        return float("nan"), float("nan")
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    return m, sd / len(d) ** 0.5
