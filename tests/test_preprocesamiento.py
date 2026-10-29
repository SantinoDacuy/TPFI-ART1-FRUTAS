"""Preprocesamiento: el codigo v1 del primer avance no cambia, la causa raiz de
tiene_tallo = 0, las mediciones nuevas y el calculo de umbrales sin fuga."""
import ast
import os

import cv2
import numpy as np
import pytest

import extraccion as X
import preprocesador as P
from conftest import RAIZ

BASE = os.path.join(RAIZ, "baseline_original")


# --- el primer avance permanece intacto ------------------------------------------------
def _fuentes(ruta):
    texto = open(ruta, encoding="utf-8").read().replace("\r\n", "\n")
    arbol = ast.parse(texto)
    out = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef):
            out[nodo.name] = ast.get_source_segment(texto, nodo)
        elif isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            out[nodo.targets[0].id] = ast.get_source_segment(texto, nodo)
    return out


def test_funciones_v1_de_preprocesador_son_las_entregadas():
    orig, act = _fuentes(os.path.join(BASE, "preprocesador.py")), _fuentes(os.path.join(RAIZ, "preprocesador.py"))
    for nombre in ("segmentar", "contorno_principal", "centrar_y_escalar", "extraer_features", "binarizar",
                   "procesar_imagen", "UMBRALES", "TAM_CANVAS"):
        assert act[nombre] == orig[nombre], f"{nombre} cambio respecto del primer avance"


def test_preprocesador_calidad_no_cambia():
    a = open(os.path.join(BASE, "preprocesador_calidad.py"), encoding="utf-8").read().replace("\r\n", "\n")
    b = open(os.path.join(RAIZ, "preprocesador_calidad.py"), encoding="utf-8").read().replace("\r\n", "\n")
    assert a == b


def test_nucleo_art1_equivale_al_entregado():
    import importlib.util
    import random
    import validacion
    spec = importlib.util.spec_from_file_location("cg_original", os.path.join(BASE, "CarGross.py"))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    from CarGross import RedART1
    rng = random.Random(0)
    for csv_ in ("dataset_morfologia.csv", "dataset_calidad_real.csv"):
        ds = validacion.leer_dataset(os.path.join(BASE, "datasets", csv_))
        for rho in (0.3, 0.55, 0.7, 0.85, 1.0):
            for k in range(3):
                idx = list(range(len(ds.entradas)))
                if k:
                    rng.shuffle(idx)
                a = v1.RedART1(len(ds.cols_features), rho, max_categorias=10 ** 6)
                b = RedART1(len(ds.cols_features), rho, max_categorias=10 ** 6)
                assert [a.clasificar(ds.entradas[i]) for i in idx] == [b.clasificar(ds.entradas[i]) for i in idx]
                assert a.top_down == b.top_down


# --- causa raiz de tiene_tallo = 0 ----------------------------------------------------------
def disco(r=60, lado=200, tallo=False, muesca=False):
    m = np.zeros((lado, lado), np.uint8)
    cv2.circle(m, (lado // 2, lado // 2 + 10), r, 255, -1)
    if tallo:
        cv2.rectangle(m, (lado // 2 - 2, lado // 2 + 10 - r - 22), (lado // 2 + 2, lado // 2 + 10 - r + 4), 255, -1)
    if muesca:
        cv2.circle(m, (lado // 2, lado // 2 + 10 - r), 18, 0, -1)
    return m


@pytest.mark.parametrize("forma", ["N14", "N4"])
def test_profundidad_defecto_v2_no_depende_de_la_version_de_opencv(monkeypatch, forma):
    original = cv2.convexityDefects

    def falso(c, h):
        d = original(c, h)
        if d is None:
            return d
        d = np.asarray(d).reshape(-1, 4)
        return d.reshape(-1, 1, 4) if forma == "N14" else d     # (N,1,4) de OpenCV 4 o (N,4) de OpenCV 5

    monkeypatch.setattr(cv2, "convexityDefects", falso)
    assert P.profundidad_defecto_v2(disco(muesca=True)) > 0.05
    assert P.profundidad_defecto_v2(disco()) < 0.05


def test_v1_queda_en_cero_cuando_opencv_devuelve_2d(monkeypatch):
    """Reproduce la causa raiz: con (N,4) la condicion ndim == 3 del primer avance falla en silencio."""
    original = cv2.convexityDefects
    monkeypatch.setattr(cv2, "convexityDefects", lambda c, h: np.asarray(original(c, h)).reshape(-1, 4))
    img = np.full((200, 200, 3), 255, np.uint8)
    img[disco(muesca=True) > 0] = (40, 60, 200)
    crudos = P.extraer_features(img, disco(muesca=True))
    assert crudos["profundidad_defecto"] == 0.0
    assert P.profundidad_defecto_v2(disco(muesca=True)) > 0.05


def test_medir_tallo_distingue_apendice_de_cuerpo_liso():
    assert P.medir_tallo(disco()) < 0.05
    assert P.medir_tallo(disco(tallo=True)) > 0.15
    assert P.medir_tallo(np.zeros((50, 50), np.uint8)) == 0.0


# --- tono circular -----------------------------------------------------------------------------
def test_tono_circular_no_penaliza_rojos_que_cruzan_el_cero():
    hsv = np.zeros((40, 40, 3), np.uint8)
    hsv[:, :, 1], hsv[:, :, 2] = 200, 200
    hsv[:, ::2, 0], hsv[:, 1::2, 0] = 2, 177        # ambos son rojos (2 y 177 estan a 5 unidades)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    masc = np.full((40, 40), 255, np.uint8)
    lineal = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]))
    est = P.estadisticas_tono(img, masc)
    assert lineal > 50 and est["desvio_tono"] < 5
    assert est["tono_medio"] < 10 or est["tono_medio"] > 350


def test_estadisticas_tono_con_mascara_vacia_o_sin_color():
    img = np.full((20, 20, 3), 128, np.uint8)       # gris: sin tono definido
    assert P.estadisticas_tono(img, np.zeros((20, 20), np.uint8))["saturacion_media"] == 0.0
    est = P.estadisticas_tono(img, np.full((20, 20), 255, np.uint8))
    assert est["desvio_tono"] == 0.0


# --- segmentacion v2 ------------------------------------------------------------------------------
def foto_sintetica():
    img = np.full((240, 320, 3), (215, 225, 230), np.uint8)
    cv2.circle(img, (160, 125), 70, (40, 60, 200), -1)
    return img


def test_segmentar_v2_aisla_el_objeto_y_es_determinista():
    img = foto_sintetica()
    m1, info = P.segmentar_v2(img, devolver_info=True)
    m2 = P.segmentar_v2(img)
    verdad = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(verdad, (160, 125), 70, 255, -1)
    iou = np.count_nonzero((m1 > 0) & (verdad > 0)) / np.count_nonzero((m1 > 0) | (verdad > 0))
    assert iou > 0.9 and (m1 == m2).all()
    assert info["candidato"] in ("lab", "grabcut", "v1") and info["puntajes"][info["candidato"]] > 0.35


def test_puntaje_de_plausibilidad_castiga_mascaras_pegadas_al_borde():
    centrada = np.zeros((100, 100), np.uint8)
    cv2.circle(centrada, (50, 50), 25, 255, -1)
    borde = np.zeros((100, 100), np.uint8)
    borde[:, :70] = 255
    assert P.puntaje_plausibilidad(centrada) > 0.6 > P.puntaje_plausibilidad(borde)
    assert P.puntaje_plausibilidad(np.zeros((100, 100), np.uint8)) == 0.0


def test_medir_foto_v1_y_v2_devuelven_todas_las_medidas():
    img = foto_sintetica()
    for seg in ("v1", "v2"):
        m = X.medir_foto(img, segmentador=seg)
        for k in ("aspecto_esferico", "textura_laplaciano", "profundidad_defecto", "solidez", "gris_promedio",
                  "desvio_hue", "area", "volumen_proxy", "defecto_score", "desvio_tono", "tono_medio",
                  "saturacion_media", "frac_oscuros", "tallo_relativo", "defecto_convexidad", "dist_tono_naranja"):
            assert k in m.crudos and np.isfinite(m.crudos[k])
    with pytest.raises(ValueError):
        X.medir_foto(img, segmentador="otro")


# --- configuracion y umbrales ----------------------------------------------------------------------------
def test_vectorizar_legado_equivale_a_binarizar_v1():
    img = foto_sintetica()
    m = X.medir_foto(img, segmentador="v1")
    v = X.vectorizar(m, X.preprocesamiento_legado("morfologia"))
    b = P.binarizar(m.crudos)
    assert v == [b[c] for c in X.preprocesamiento_legado("morfologia")["cols_features"]]


def crudos_falsos(valores):
    base = {k: 0.0 for k in ("aspecto_esferico", "textura_laplaciano", "profundidad_defecto", "solidez",
                             "gris_promedio", "desvio_hue", "area", "volumen_proxy", "defecto_score",
                             "desvio_tono", "saturacion_media", "tallo_relativo", "defecto_convexidad",
                             "frac_oscuros", "dist_tono_naranja", "tono_medio")}
    return [{**base, "aspecto_esferico": v, "area": v * 1000, "textura_laplaciano": v * 10} for v in valores]


def test_umbrales_por_mediana_usan_solo_los_datos_recibidos():
    a = X.umbrales_desde_entrenamiento("calidad", crudos_falsos([1, 2, 3, 4, 5]), "mediana")
    b = X.umbrales_desde_entrenamiento("calidad", crudos_falsos([10, 20, 30, 40, 50]), "mediana")
    assert a["es_grande"] == 3000 and b["es_grande"] == 30000        # mediana exacta, sin redondeo
    assert a["es_grande"] != X.preprocesamiento_legado("calidad")["umbrales"]["es_grande"]


def test_umbrales_fijos_no_dependen_de_los_datos():
    a = X.umbrales_desde_entrenamiento("morfologia", crudos_falsos([1, 2, 3]), "fijo", tallo="v1", tono="lineal")
    b = X.umbrales_desde_entrenamiento("morfologia", crudos_falsos([9, 9, 9]), "fijo", tallo="v1", tono="lineal")
    assert a == b and a["es_esferica"] == 0.8 and a["tiene_tallo"] == 0.06


def test_estrategia_desconocida_y_umbral_faltante():
    with pytest.raises(ValueError, match="estrategia desconocida"):
        X.umbrales_desde_entrenamiento("calidad", crudos_falsos([1, 2]), "azar")
    with pytest.raises(ValueError, match="faltan umbrales"):
        X.configuracion("calidad", {"es_grande": 1.0})


def test_otsu_separa_dos_grupos():
    vals = sorted([0.1, 0.12, 0.11, 0.13, 0.9, 0.92, 0.88, 0.91])
    assert 0.13 <= X._corte_otsu(vals) <= 0.9


def test_configuracion_ordena_extras_segun_el_esquema():
    um = {c: 1.0 for c in ("es_esferica", "cascara_rugosa", "tiene_tallo", "densidad_alta",
                          "pigmentacion_oscura", "color_uniforme", "tono_calido", "saturacion_alta")}
    pre = X.configuracion("morfologia", um, extras=("saturacion_alta", "tono_calido"))
    assert pre["cols_features"][-2:] == ["tono_calido", "saturacion_alta"]
    assert pre["version"] == "v2" and set(pre["reglas"]) == set(pre["umbrales"])


def test_binarizar_respeta_operadores():
    pre = X.preprocesamiento_legado("calidad")
    crudos = {"area": 20000.0, "volumen_proxy": 1.0, "textura_laplaciano": 1.0, "defecto_score": 0.5, "desvio_hue": 1.0}
    assert X.binarizar(crudos, pre) == {"es_grande": 1, "es_pesada": 0, "textura_firme": 1,
                                        "sin_defectos_superficie": 0, "color_homogeneo": 1}
