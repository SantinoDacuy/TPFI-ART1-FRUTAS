"""Segunda ronda: informe reservado sin fuga, opciones de entrenamiento, descriptores cromaticos nuevos,
tope de categorias, comite experimental y diagnostico de robustez."""
import os
import sys
from collections import defaultdict
from dataclasses import replace

import cv2
import numpy as np
import pytest

import extraccion as X
import preprocesador as P
from conftest import RAIZ
from evaluacion import banco, cadena, experimentos as E

sys.path.insert(0, os.path.join(RAIZ, "scripts"))
MANIFIESTO = os.path.join(RAIZ, "datasets_v2", "manifiesto_imagenes.csv")
MEDICIONES = os.path.join(RAIZ, "resultados", "mediciones")
datos = pytest.mark.skipif(not (os.path.isfile(MANIFIESTO) and os.path.isfile(os.path.join(MEDICIONES, "mediciones_v1.csv"))),
                           reason="faltan datasets_v2/ o resultados/mediciones/")


def fotos_falsas(n_grupos=14, por_grupo=3):
    out = []
    for e in ("manzana_good", "naranja"):
        for g in range(n_grupos):
            for k in range(1 + g % por_grupo):
                out.append({"foto_id": f"{e}{g}_{k}", "estrato": e, "grupo": f"{e}{g}", "tipo_real": "carozo" if "manz" in e else "citrico",
                            "calidad_real": "premium" if "manz" in e else ""})
    return out


# --- particion de informe -------------------------------------------------------------------------
def test_reservar_informe_por_grupos_estratificado_y_determinista():
    a = banco.reservar_informe([dict(f) for f in fotos_falsas()], 3)
    b = banco.reservar_informe([dict(f) for f in fotos_falsas()], 3)
    assert [f["bloque"] for f in a] == [f["bloque"] for f in b]
    por_grupo = defaultdict(set)
    for f in a:
        por_grupo[f["grupo"]].add(f["bloque"])
    assert all(len(v) == 1 for v in por_grupo.values())                      # un grupo no se parte
    for e in ("manzana_good", "naranja"):
        bl = {f["bloque"] for f in a if f["estrato"] == e}
        assert bl == set(banco.BLOQUES)


def test_el_informe_es_identico_en_todas_las_semillas_internas_y_no_se_mezcla_con_desarrollo():
    base = banco.reservar_informe([dict(f) for f in fotos_falsas()], 5)
    informe = {f["foto_id"] for f in base if f["bloque"] == "informe"}
    vistos = []
    for s in range(6):
        fs = E.particion_informe(fotos_falsas(), s, 5)
        assert {f["foto_id"] for f in fs if f["particion"] == "prueba"} == informe
        assert not any(f["particion"] == "prueba" for f in fs if f["bloque"] == "desarrollo")
        assert {f["particion"] for f in fs if f["bloque"] == "desarrollo"} == {"entrenamiento", "validacion"}
        vistos.append(tuple(f["particion"] for f in fs))
    assert len(set(vistos)) > 1                                              # las particiones internas SI varian


@datos
def test_manifiesto_ronda2_declara_bloque_y_conserva_trazabilidad():
    filas = banco.leer_manifiesto_ronda2(MANIFIESTO) if hasattr(banco, "leer_manifiesto_ronda2") else None
    import csv
    with open(MANIFIESTO, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert set(banco.COLUMNAS_MANIFIESTO_RONDA2) <= set(filas[0]) and "particion" not in filas[0]
    assert {f["bloque"] for f in filas} == set(banco.BLOQUES) and len(filas) == 339
    por_grupo = defaultdict(set)
    for f in filas:
        por_grupo[f["grupo"]].add(f["bloque"])
    assert all(len(v) == 1 for v in por_grupo.values())


# --- ninguna decision usa el informe ---------------------------------------------------------------------
@datos
def test_el_informe_no_interviene_en_umbrales_ni_en_rho():
    from busqueda import SEMILLA_INFORME
    import busqueda as B
    fotos = banco.inventariar(os.path.join(RAIZ, "datasets", "processed_images_fruits")) \
        if os.path.isdir(os.path.join(RAIZ, "datasets", "processed_images_fruits")) else None
    if fotos is None:
        import csv
        with open(MANIFIESTO, newline="", encoding="utf-8") as f:
            fotos = list(csv.DictReader(f))
    med = E.leer_mediciones(os.path.join(MEDICIONES, "mediciones_v1.csv"))
    var = replace(B.COMPATIBLE, extras_morf=("croma_alto",), extras_cal=("croma_alto",))
    base = E.correr_split(var, fotos, med, 2, grilla=(0.5, 0.6, 0.7), semilla_informe=SEMILLA_INFORME)
    fs = E.particion_informe(fotos, 2, SEMILLA_INFORME)
    alterado = {k: dict(v) for k, v in med.items()}
    for f in fs:
        if f["particion"] == "prueba":
            for m in alterado[f["foto_id"]]:
                alterado[f["foto_id"]][m] *= 1e3
    otro = E.correr_split(var, fotos, alterado, 2, grilla=(0.5, 0.6, 0.7), semilla_informe=SEMILLA_INFORME)
    for etapa in ("morfologia", "calidad"):
        assert otro["_pre"][etapa]["umbrales"] == base["_pre"][etapa]["umbrales"]
        assert otro["_modelos"][etapa].red.top_down == base["_modelos"][etapa].red.top_down
        assert otro[f"rho_{etapa}"] == base[f"rho_{etapa}"]


# --- opciones de la variante -----------------------------------------------------------------------------------
def test_reordenar_canonico_es_determinista_y_no_usa_etiquetas():
    v = [[1, 1, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1]]
    a = E._reordenar(v, list("abcde"), "canonico_asc", list("12345"))
    b = E._reordenar(list(reversed(v)), list("edcba"), "canonico_asc", list("54321"))
    assert a[0] == b[0] and a[2] == b[2]                                      # mismo orden aunque llegue permutado
    assert [sum(x) for x in a[0]] == sorted(sum(x) for x in v)
    with pytest.raises(ValueError):
        E._reordenar(v, list("abcde"), "raro", list("12345"))


def test_tope_de_categorias_filtra_rho_admisibles():
    import random
    rng = random.Random(0)
    entr = [[rng.randint(0, 1) for _ in range(6)] for _ in range(60)]
    et = [rng.choice("ab") for _ in entr]
    pre = X.preprocesamiento_legado("morfologia")
    fotos = [{"foto_id": str(i), "tipo_real": e, "calidad_real": ""} for i, e in enumerate(et)]
    vec = {str(i): v for i, v in enumerate(entr)}
    libre = E.Variante("x", legado=True)
    rho, tabla = E.elegir_rho(libre, "morfologia", pre, entr, et, vec, fotos)
    k_por_rho = {t["rho"]: t["k"] for t in tabla if t["ok"]}
    tope = sorted(k_por_rho.values())[2]
    rho2, _ = E.elegir_rho(replace(libre, tope_k=tope), "morfologia", pre, entr, et, vec, fotos)
    assert k_por_rho[rho2] <= tope
    rho3, _ = E.elegir_rho(replace(libre, tope_k=0), "morfologia", pre, entr, et, vec, fotos)   # inalcanzable: toma el de menos categorias
    assert k_por_rho[rho3] == min(k_por_rho.values())


def test_soporte_minimo_deja_sin_etiqueta_las_categorias_chicas():
    from CarGross import RedART1, construir_mapeo
    red = RedART1(4, 0.9)
    red.top_down = [[1, 1, 0, 0], [0, 0, 1, 1]]
    X_ = [[1, 1, 0, 0]] * 3 + [[0, 0, 1, 1]]
    m, _ = construir_mapeo(red, X_, ["a", "a", "a", "b"], soporte_min=2)
    assert m["0"]["etiqueta"] == "a" and m["1"]["etiqueta"] is None and m["1"]["estado"] == "sin_soporte"
    assert m["1"]["etiqueta_candidata"] == "b"


def test_derivacion_minima_exige_coincidencia_de_la_etapa_1():
    from CarGross import Modelo, RedART1
    r1 = RedART1(4, 0.5)
    r1.top_down = [[1, 1, 0, 0]]
    m1 = Modelo(r1, {"cols_features": list("abcd"), "mapeo_clusters": {"0": {"n": 2, "etiqueta": "carozo", "estado": "etiquetada", "pureza": 1.0}}}, None, 2, [], None)
    r2 = RedART1(2, 0.5)
    r2.top_down = [[1, 0]]
    m2 = Modelo(r2, {"cols_features": list("ab"), "mapeo_clusters": {"0": {"n": 2, "etiqueta": "premium", "estado": "etiquetada", "pureza": 1.0}}}, None, 2, [], None)
    fotos = [{"foto_id": "x", "tipo_real": "carozo", "calidad_real": "premium", "estrato": "g", "grupo": "1"}]
    vm, vc = {"x": [1, 1, 1, 1]}, {"x": [1, 0]}                              # coincidencia = 2/4 = 0.5
    assert cadena.evaluar_cadena(fotos, vm, vc, m1, m2)[0]["derivada"] is True
    assert cadena.evaluar_cadena(fotos, vm, vc, m1, m2, derivacion_min=0.75)[0]["derivada"] is False


def test_comite_vota_etiquetas_y_no_modifica_pesos():
    import random
    rng = random.Random(1)
    centros = {"a": [1, 1, 1, 0, 0, 0], "b": [0, 0, 0, 1, 1, 1]}
    vec, et = [], []
    for _ in range(40):
        c = rng.choice("ab")
        vec.append([b if rng.random() > 0.1 else 1 - b for b in centros[c]])
        et.append(c)
    pre = X.preprocesamiento_legado("morfologia")
    var = E.Variante("c", legado=True, comite=5, desempate="mayoritaria_global")
    m, info = E.entrenar_etapa(var, "morfologia", vec, et, 0.7, pre)
    assert isinstance(m, E.ModeloComite) and len(info["k_miembros"]) == 5
    h = m.red.huella()
    ev = m.red.evaluar(centros["a"])
    assert ev["resuena"] and ev["ganador"] == "a" and m.categoria(ev["ganador"])["etiqueta"] == "a"
    assert m.red.huella() == h


# --- descriptores cromaticos nuevos ------------------------------------------------------------------------------------
def disco_color(bgr, fondo=(215, 225, 230)):
    img = np.full((200, 200, 3), fondo, np.uint8)
    cv2.circle(img, (100, 100), 60, bgr, -1)
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 60, 255, -1)
    return img, m


def test_fracciones_de_tono_identifican_el_color_dominante():
    naranja, m = disco_color((0, 128, 255))
    verde, _ = disco_color((40, 200, 40))
    f_n, f_v = P.estadisticas_color(naranja, m), P.estadisticas_color(verde, m)
    assert f_n["frac_naranja"] > 0.9 and f_n["frac_verde"] < 0.05 and f_v["frac_verde"] > 0.9
    assert abs(sum(f_n[k] for k in P.FAMILIAS_TONO) - 1.0) < 1e-6 or sum(f_n[k] for k in P.FAMILIAS_TONO) <= 1.0


def test_croma_y_gradiente_distinguen_color_vivo_y_uniforme_de_apagado_y_degradado():
    vivo, m = disco_color((0, 128, 255))
    gris, _ = disco_color((128, 128, 128))
    assert P.estadisticas_color(vivo, m)["croma_medio"] > 40 > P.estadisticas_color(gris, m)["croma_medio"]
    degradado = vivo.copy()
    degradado[:, :, 1] = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    assert P.estadisticas_color(degradado, m)["gradiente_color"] > P.estadisticas_color(vivo, m)["gradiente_color"]


def test_balance_de_blancos_corrige_un_tinte_global_y_el_nucleo_recorta_el_borde():
    img, m = disco_color((128, 128, 128), fondo=(128, 128, 128))
    tinte = np.clip(img.astype(np.float32) * np.array([0.6, 1.0, 1.4]), 0, 255).astype(np.uint8)
    corregida = P.aplicar_factores(tinte, P.factores_gris(tinte))
    assert np.abs(corregida.astype(int) - img.astype(int)).max() <= 3
    assert cv2.countNonZero(P.nucleo_mascara(m)) < cv2.countNonZero(m)
    assert cv2.countNonZero(P.nucleo_mascara(np.zeros_like(m))) == 0


def test_extras_nuevos_se_configuran_con_cada_opcion_de_color():
    for etapa, esq_extras in (("morfologia", ("croma_alto", "naranja_dominante", "color_con_gradiente")),
                              ("calidad", ("croma_alto", "color_sin_gradiente"))):
        for color, suf in X.COLOR_SUFIJOS.items():
            um = X.umbrales_desde_entrenamiento(etapa, [dict.fromkeys(X.val.MEDIDAS_CONOCIDAS, 1.0)] * 3, "compromiso",
                                                tallo="convexidad", tono="circular", extras=esq_extras, color=color)
            pre = X.configuracion(etapa, um, "v1", tallo="convexidad", tono="circular", extras=esq_extras, color=color)
            assert pre["opciones"]["color"] == color
            assert all(pre["reglas"][c]["crudo"].endswith(suf) or suf == "" for c in esq_extras)


def test_estrategia_compromiso_conserva_los_cortes_originales_de_morfologia():
    crudos = [dict.fromkeys(X.val.MEDIDAS_CONOCIDAS, v) for v in (1.0, 2.0, 3.0)]
    um = X.umbrales_desde_entrenamiento("morfologia", crudos, "compromiso", tallo="convexidad", tono="circular")
    assert um == {"es_esferica": 0.8, "cascara_rugosa": 150.0, "tiene_tallo": 0.06, "densidad_alta": 0.9,
                  "pigmentacion_oscura": 130.0, "color_uniforme": 12.0}
    um_c = X.umbrales_desde_entrenamiento("calidad", crudos, "compromiso", tono="circular")
    assert um_c["es_grande"] == 2.0                                            # calidad: mediana del entrenamiento (lote)
    um_p = X.umbrales_desde_entrenamiento("morfologia", crudos, "compromiso", tallo="protuberancia", tono="circular")
    assert um_p["tiene_tallo"] == 2.0 and um_p["es_esferica"] == 0.8          # detector nuevo: sin corte original
    um_m = X.umbrales_desde_entrenamiento("morfologia", crudos, "mediana", tallo="convexidad", tono="circular")
    assert um_m["es_esferica"] == 2.0                                          # experimental: cambia el significado


# --- robustez ---------------------------------------------------------------------------------------------------------
def test_sustitucion_de_fondo_y_perturbaciones_de_iluminacion():
    import robustez as R
    img, m = disco_color((40, 60, 200))
    nuevo = R.sustituir_fondo(img, m, (50, 140, 60))
    assert tuple(nuevo[5, 5]) == (50, 140, 60) and tuple(nuevo[100, 100]) == (40, 60, 200)      # fondo cambia, fruta no
    assert R.ILUMINACION["luz_oscura"](img).mean() < img.mean() < R.ILUMINACION["luz_clara"](img).mean()
    assert R.ILUMINACION["tinte_calido"](img)[0, 0, 2] > img[0, 0, 2] or img[0, 0, 2] == 255
