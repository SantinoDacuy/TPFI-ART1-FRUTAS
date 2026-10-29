"""Evaluacion: grupos y particiones sin fuga, metricas con no-respuestas, cadena de
dos etapas y regresion contra las cifras de la linea de base."""
import os

import pytest

import extraccion as X
from CarGross import Modelo, RedART1
from conftest import RAIZ
from evaluacion import banco, cadena, experimentos as E, metricas

MANIFIESTO = os.path.join(RAIZ, "datasets_v2", "manifiesto_imagenes.csv")
MEDICIONES = os.path.join(RAIZ, "resultados", "mediciones")
necesita_datos = pytest.mark.skipif(
    not (os.path.isfile(MANIFIESTO) and os.path.isfile(os.path.join(MEDICIONES, "mediciones_v1.csv"))),
    reason="faltan datasets_v2/ o resultados/mediciones/ (se generan con los scripts)")


# --- banco: grupos y particiones ----------------------------------------------------------
def test_clave_de_tiempo_por_nombre():
    assert banco._clave_tiempo("IMG20200728175856.jpg")[0] == "t"
    assert banco._clave_tiempo("20190809_115439.jpg")[0] == "t"
    assert banco._clave_tiempo("IMG_8536.JPG") == ("c", 8536)
    assert banco._clave_tiempo("foto.jpg") is None
    a, b = banco._clave_tiempo("IMG20200728175907.jpg"), banco._clave_tiempo("IMG20200728175908.jpg")
    assert b[1] - a[1] == 1
    assert banco._clave_tiempo("IMG20200728175908_01.jpg") == b      # copia editada = misma toma


def fotos_falsas():
    nombres = ["IMG20200101100000.jpg", "IMG20200101100001.jpg", "IMG20200101100002.jpg",   # rafaga
               "IMG20200101100030.jpg", "IMG20200101100031.jpg",                             # otra
               "IMG20200101100100.jpg", "IMG_0100.JPG", "IMG_0101.JPG", "IMG_0105.JPG", "x.jpg"]
    return [{"foto_id": f"F{i}", "estrato": "a", "tipo_real": "t", "calidad_real": "", "_clave": banco._clave_tiempo(n)}
            for i, n in enumerate(nombres)]


def test_asignar_grupos_por_rafaga_y_contador():
    f = banco.asignar_grupos(fotos_falsas(), gap_s=3, gap_contador=1)
    g = [x["grupo"] for x in f]
    assert g[0] == g[1] == g[2] != g[3] == g[4] != g[5]
    assert g[6] == g[7] != g[8]                    # contadores consecutivos se agrupan, uno lejano no
    assert len(set(g)) == len(g) - 4 - 1 + 1 or g[9] not in g[:9]      # la foto sin clave forma su propio grupo


def test_particion_por_grupo_sin_compartir_y_determinista():
    fotos = []
    for e in ("x", "y"):
        for gi in range(12):
            for k in range(1 + gi % 4):
                fotos.append({"foto_id": f"{e}{gi}_{k}", "estrato": e, "grupo": f"{e}{gi}", "tipo_real": "t", "calidad_real": ""})
    a = banco.particionar([dict(f) for f in fotos], 3)
    b = banco.particionar([dict(f) for f in fotos], 3)
    c = banco.particionar([dict(f) for f in fotos], 4)
    assert [f["particion"] for f in a] == [f["particion"] for f in b]
    assert [f["particion"] for f in a] != [f["particion"] for f in c]
    por_grupo = {}
    for f in a:
        por_grupo.setdefault(f["grupo"], set()).add(f["particion"])
    assert all(len(v) == 1 for v in por_grupo.values())                    # ningun grupo en dos particiones
    for e in ("x", "y"):
        assert {f["particion"] for f in a if f["estrato"] == e} == set(banco.PARTICIONES)


@necesita_datos
def test_manifiesto_es_consistente():
    filas = banco.leer_manifiesto(MANIFIESTO)
    assert len(filas) == 339 and len({f["foto_id"] for f in filas}) == 339
    campo = "bloque" if "bloque" in filas[0] else "particion"
    por_grupo = {}
    for f in filas:
        por_grupo.setdefault(f["grupo"], set()).add(f[campo])
    assert all(len(v) == 1 for v in por_grupo.values())
    assert sum(f["tipo_real"] == "carozo" for f in filas) == 219
    assert all((f["calidad_real"] != "") == (f["tipo_real"] == "carozo") for f in filas)


@necesita_datos
def test_el_manifiesto_de_la_ronda_1_se_conserva():
    ronda1 = os.path.join(RAIZ, "datasets_v2", "ronda1_medianas", "manifiesto_imagenes_ronda1.csv")
    if not os.path.isfile(ronda1):
        pytest.skip("no hay respaldo de la ronda 1")
    filas = banco.leer_manifiesto(ronda1)
    assert len(filas) == 339 and "particion" in filas[0]


# --- metricas -------------------------------------------------------------------------------
def test_no_respuestas_cuentan_como_error_y_se_informan():
    v = ["a", "a", "b", "b"]
    p = ["a", None, "b", "a"]
    assert metricas.exactitud(v, p) == 0.5
    conf = metricas.matriz_confusion(v, p, ["a", "b"])
    assert conf["a"] == {"a": 1, metricas.SIN_ETIQUETA: 1} and conf["b"] == {"b": 1, "a": 1}
    f = metricas.f1_por_clase(v, p, ["a", "b"])
    assert f["a"]["precision"] == 0.5 and f["a"]["recall"] == 0.5 and f["b"]["precision"] == 1.0


def test_f1_macro_ignora_clases_sin_soporte_y_linea_base_mayoritaria():
    assert metricas.f1_macro(["a", "a"], ["a", "a"], ["a", "b"]) == 1.0
    lb = metricas.linea_base_mayoritaria(["a", "a", "a", "b"], "a", ["a", "b"])
    assert lb["exactitud"] == 0.75 and lb["f1_macro"] == pytest.approx((0.857142857 + 0) / 2)
    assert metricas.clase_mayoritaria(["b", "a", "a", "b"]) == "a"          # empate: orden alfabetico


def test_bootstrap_por_grupos_es_reproducible_y_acotado():
    ok = [True] * 8 + [False] * 4
    g = [str(i // 2) for i in range(12)]
    a = metricas.ic_bootstrap_por_grupos(ok, g, repeticiones=300, semilla=1)
    assert a == metricas.ic_bootstrap_por_grupos(ok, g, repeticiones=300, semilla=1)
    assert 0 <= a[0] <= 2 / 3 <= a[1] <= 1
    assert all(x != x for x in metricas.ic_bootstrap_por_grupos([True], ["a"]))   # un solo grupo: sin IC


# --- cadena de dos etapas ---------------------------------------------------------------------------
def modelo(prototipos, etiquetas, n):
    red = RedART1(n, 0.6)
    red.top_down = prototipos
    mapeo = {str(i): {"n": 1, "etiqueta": e, "estado": "etiquetada" if e else "empate", "pureza": 1.0}
             for i, e in enumerate(etiquetas)}
    return Modelo(red, {"cols_features": [f"v{i}" for i in range(n)], "mapeo_clusters": mapeo}, None, 2, [], None)


def test_cadena_deriva_solo_carozo_y_cuenta_cada_error():
    m1 = modelo([[1, 1, 0], [0, 0, 1], [0, 1, 1]], ["carozo", "citrico", None], 3)
    m2 = modelo([[1, 0], [0, 1]], ["premium", "descarte"], 2)
    fotos = [
        {"foto_id": "a", "tipo_real": "carozo", "calidad_real": "premium", "estrato": "g", "grupo": "1"},
        {"foto_id": "b", "tipo_real": "carozo", "calidad_real": "descarte", "estrato": "g", "grupo": "2"},
        {"foto_id": "c", "tipo_real": "citrico", "calidad_real": "", "estrato": "n", "grupo": "3"},
        {"foto_id": "d", "tipo_real": "citrico", "calidad_real": "", "estrato": "n", "grupo": "4"},   # derivada por error
        {"foto_id": "e", "tipo_real": "tropical", "calidad_real": "", "estrato": "t", "grupo": "5"},  # categoria sin etiqueta
        {"foto_id": "f", "tipo_real": "carozo", "calidad_real": "premium", "estrato": "g", "grupo": "6"},  # novedad
    ]
    vm = {"a": [1, 1, 0], "b": [1, 1, 0], "c": [0, 0, 1], "d": [1, 1, 0], "e": [0, 1, 1], "f": [1, 0, 1]}
    vc = {"a": [1, 0], "b": [1, 0], "c": [0, 1], "d": [0, 1], "e": [0, 1], "f": [1, 0]}
    huellas = (m1.red.huella(), m2.red.huella())
    regs = cadena.evaluar_cadena(fotos, vm, vc, m1, m2)
    assert (m1.red.huella(), m2.red.huella()) == huellas                    # inferencia pura
    r = cadena.resumir_cadena(regs, "carozo", "premium")
    assert r["manzanas_derivadas"] == 2 and r["no_manzanas_derivadas"] == 1
    assert r["e1_novedades"] == 1 and r["e1_sin_etiqueta"] == 1
    assert r["e1_tipo_correcto"] == 3                                      # a, b, c
    assert r["cadena_manzanas_correctas"] == 1 and r["cadena_total_correcta"] == 2   # a; a + c
    assert r["e2_aislada_n"] == 3 and r["e2_aislada_exactitud"] == pytest.approx(2 / 3)
    assert r["linea_base_tipo"]["exactitud"] == 0.5
    assert [x["morf_estado"] for x in regs][4] == "empate"


# --- regresion contra la linea de base y ausencia de fuga ---------------------------------------------------
@necesita_datos
def test_pipeline_con_mediciones_reproduce_la_linea_base_entregada():
    from CarGross import cargar_modelo
    fotos = banco.leer_manifiesto(MANIFIESTO)
    med = E.leer_mediciones(os.path.join(MEDICIONES, "mediciones_v1.csv"))
    mm = cargar_modelo(os.path.join(RAIZ, "modelos", "art1_morfologia.json"))
    mc = cargar_modelo(os.path.join(RAIZ, "modelos", "art1_calidad.json"))
    pm, pc = X.preprocesamiento_legado("morfologia"), X.preprocesamiento_legado("calidad")
    vm = {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pm) for f in fotos}
    vc = {f["foto_id"]: X.vectorizar(med[f["foto_id"]], pc) for f in fotos}
    r = cadena.resumir_cadena(cadena.evaluar_cadena(fotos, vm, vc, mm, mc))
    assert (r["e1_tipo_correcto"], r["manzanas_derivadas"], r["cadena_manzanas_correctas"],
            r["no_manzanas_derivadas"]) == (167, 96, 65, 44)


@necesita_datos
def test_no_hay_fuga_de_prueba_ni_de_validacion_en_los_umbrales():
    fotos = banco.leer_manifiesto(MANIFIESTO)
    med = E.leer_mediciones(os.path.join(MEDICIONES, "mediciones_v1.csv"))
    var = E.Variante("t", segmentador="v1", tallo="v1", tono="lineal", umbral_morf="mediana", umbral_cal="mediana",
                     extras_morf=("tono_calido",), rho_morf=0.7, rho_cal=0.6)
    base = E.correr_split(var, fotos, med, 0)
    por = {f["foto_id"]: f["particion"] for f in E.particion_con_semilla(fotos, 0)}
    alterado = {fid: dict(v) for fid, v in med.items()}
    for fid, part in por.items():
        if part != "entrenamiento":
            for k in alterado[fid]:
                alterado[fid][k] *= 1000.0                # otra escala: solo debe cambiar la evaluacion
    otro = E.correr_split(var, fotos, alterado, 0)
    for etapa in ("morfologia", "calidad"):
        assert otro["_pre"][etapa]["umbrales"] == base["_pre"][etapa]["umbrales"]
        assert otro["_modelos"][etapa].red.top_down == base["_modelos"][etapa].red.top_down
    assert otro["entrenamiento"]["e1_exactitud"] == base["entrenamiento"]["e1_exactitud"]


@necesita_datos
def test_rho_se_elige_solo_con_validacion():
    fotos = banco.leer_manifiesto(MANIFIESTO)
    med = E.leer_mediciones(os.path.join(MEDICIONES, "mediciones_v1.csv"))
    var = E.Variante("t", segmentador="v1", tallo="v1", tono="lineal", umbral_morf="mediana", umbral_cal="mediana")
    a = E.correr_split(var, fotos, med, 1, grilla=(0.5, 0.6, 0.7, 0.8))
    alterado = {fid: dict(v) for fid, v in med.items()}
    for f in E.particion_con_semilla(fotos, 1):
        if f["particion"] == "prueba":
            for k in alterado[f["foto_id"]]:
                alterado[f["foto_id"]][k] = 0.0
    b = E.correr_split(var, fotos, alterado, 1, grilla=(0.5, 0.6, 0.7, 0.8))
    assert (a["rho_morfologia"], a["rho_calidad"]) == (b["rho_morfologia"], b["rho_calidad"])
    assert a["barrido_morfologia"] == b["barrido_morfologia"]
