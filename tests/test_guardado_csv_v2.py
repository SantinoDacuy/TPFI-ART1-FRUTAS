"""Guardado de modelos desde un CSV v2 (variables adicionales): el preprocesamiento exacto que produjo el CSV
viaja con el modelo; nunca se supone que fue el del primer avance."""
import csv
import json
import os

import cv2
import numpy as np
import pytest

import CarGross
import extraccion as X
import inferencia
import probar_imagen
import validacion as val
from CarGross import cargar_modelo
from conftest import RAIZ
from evaluacion import banco, experimentos as E
from validacion import COD_ENTRADA, COD_OK, COD_USO

MANIFIESTO = os.path.join(RAIZ, "datasets_v2", "manifiesto_imagenes.csv")
MEDICIONES = os.path.join(RAIZ, "resultados", "mediciones", "mediciones_v1.csv")
pytestmark = pytest.mark.skipif(not (os.path.isfile(MANIFIESTO) and os.path.isfile(MEDICIONES)),
                                reason="faltan datasets_v2/manifiesto_imagenes.csv o resultados/mediciones/")

EXTRAS_CAL = ("sin_oscurecimiento",)


@pytest.fixture
def v2(tmp_path):
    """CSV de calidad v2 (6 variables) + su preprocesamiento (umbrales de las manzanas de la mitad de las fotos)."""
    fotos = [f for f in banco.leer_manifiesto(MANIFIESTO) if f["tipo_real"] == "carozo"][::2]
    med = E.leer_mediciones(MEDICIONES)
    um = X.umbrales_desde_entrenamiento("calidad", [med[f["foto_id"]] for f in fotos], "mediana",
                                        tallo="convexidad", tono="circular", extras=EXTRAS_CAL)
    pre = X.configuracion("calidad", um, segmentador="v1", tallo="convexidad", tono="circular", extras=EXTRAS_CAL,
                          origen_umbrales="mediana de las manzanas de entrenamiento (prueba)")
    csv_ = tmp_path / "calidad_entrenamiento.csv"
    with open(csv_, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fruta_id"] + pre["cols_features"] + ["calidad_real"])
        for fo in fotos:
            w.writerow([fo["foto_id"]] + X.vectorizar(med[fo["foto_id"]], pre) + [fo["calidad_real"]])
    X.escribir_preprocesamiento_csv(pre, str(tmp_path / "preprocesamiento_calidad.csv"))
    return {"csv": str(csv_), "pre": pre, "dir": tmp_path, "med": med, "fotos": fotos}


def correr(argv, capsys):
    rc = CarGross.main(argv)
    o = capsys.readouterr()
    return rc, o.out, o.err


def foto_sintetica(ruta):
    img = np.full((240, 320, 3), (215, 225, 230), np.uint8)
    cv2.circle(img, (160, 125), 70, (40, 60, 200), -1)
    cv2.imwrite(str(ruta), img)
    return img


def test_el_csv_v2_tiene_seis_variables_y_el_preprocesamiento_tambien(v2):
    ds = val.leer_dataset(v2["csv"])
    assert ds.cols_features[-1] == "sin_oscurecimiento" and len(ds.cols_features) == 6
    assert v2["pre"]["cols_features"] == ds.cols_features and set(v2["pre"]["reglas"]) == set(ds.cols_features)


def test_guardar_desde_csv_v2_conserva_reglas_umbrales_orden_y_procedencia(v2, capsys):
    destino = str(v2["dir"] / "m.json")
    rc, out, err = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino], capsys)
    assert rc == COD_OK and "preprocesamiento tomado de" in out and "tipo fotografias" in out
    d = json.load(open(destino, encoding="utf-8"))
    assert d["tipo_modelo"] == "fotografias"
    assert d["preprocesamiento"] == json.loads(json.dumps(v2["pre"]))              # reglas, umbrales exactos, orden
    assert d["preprocesamiento"]["reglas"]["sin_oscurecimiento"] == {"crudo": "frac_oscuros", "op": "<="}
    assert d["metadata_extra"]["cols_features"] == v2["pre"]["cols_features"]
    assert d["metadata_extra"]["preprocesamiento_origen"] == "preprocesamiento_calidad.csv"   # procedencia
    assert d["metadata_extra"]["entrenamiento"]["csv_sha256"]
    m = cargar_modelo(destino)
    assert m.tipo == "fotografias" and not m.legado and len(m.preprocesamiento["cols_features"]) == 6


def test_predicciones_identicas_antes_y_despues_de_guardar(v2, tmp_path, capsys):
    ds = val.leer_dataset(v2["csv"])
    red, meta = CarGross.entrenar_desde_dataset(ds, 0.6, epocas=3)
    ruta = str(tmp_path / "m2.json")
    red.guardar(ruta, meta, preprocesamiento=v2["pre"])
    m = cargar_modelo(ruta)
    for x in ds.entradas:
        a, b = red.evaluar(x), m.red.evaluar(x)
        assert (a["ganador"], a["resuena"], a["ratio"]) == (b["ganador"], b["resuena"], b["ratio"])
        assert m.categoria(b["ganador"])["etiqueta"] == (
            meta["mapeo_clusters"][str(a["ganador"])]["etiqueta"] if a["resuena"] else m.categoria(None)["etiqueta"])


def test_una_fotografia_produce_la_misma_representacion_que_el_csv(v2, tmp_path, capsys):
    destino = str(v2["dir"] / "cal.json")
    assert correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino], capsys)[0] == COD_OK
    img = foto_sintetica(tmp_path / "f.png")
    m = cargar_modelo(destino)
    # misma representacion: el vector que arma inferencia desde el modelo guardado == el del preprocesamiento original
    med = X.medir_foto(img, segmentador=m.preprocesamiento["segmentador"])
    assert X.vectorizar(med, m.preprocesamiento) == X.vectorizar(med, v2["pre"])
    # y la cadena completa usa los modelos: se arma una etapa 1 propia con el preprocesamiento de la etapa 1
    fotos = banco.leer_manifiesto(MANIFIESTO)
    medall = v2["med"]
    pm = X.configuracion("morfologia", X.umbrales_desde_entrenamiento("morfologia", [medall[f["foto_id"]] for f in fotos],
                                                                      "compromiso", tallo="convexidad", tono="circular"),
                         segmentador="v1", tallo="convexidad", tono="circular")
    rm = CarGross.RedART1(len(pm["cols_features"]), 0.8)
    vec = X.vectorizar(X.medir_foto(img, segmentador="v1"), pm)
    rm.top_down = [list(vec)]
    ruta_m = str(tmp_path / "morf.json")
    rm.guardar(ruta_m, {"cols_features": pm["cols_features"],
                        "mapeo_clusters": {"0": {"n": 3, "etiqueta": "carozo", "estado": "etiquetada", "pureza": 1.0}}},
               preprocesamiento=pm)
    r = inferencia.inferir_imagen(img, cargar_modelo(ruta_m), m)
    e2 = r["etapa2"]
    assert r["derivada"] and e2["cols_features"] == v2["pre"]["cols_features"]
    assert e2["vector"] == X.vectorizar(med, v2["pre"])                  # exactamente la representacion del CSV
    for col in e2["cols_features"]:
        assert e2["umbrales"][col] == v2["pre"]["umbrales"][col]


def test_sin_preprocesamiento_se_guarda_un_modelo_de_vectores_y_no_procesa_fotos(v2, tmp_path, capsys):
    os.remove(v2["dir"] / "preprocesamiento_calidad.csv")
    destino = str(tmp_path / "vec.json")
    rc, out, _ = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino], capsys)
    assert rc == COD_OK and "VECTORES" in out and "tipo vectores" in out
    m = cargar_modelo(destino)
    assert m.tipo == "vectores" and m.preprocesamiento is None
    foto_sintetica(tmp_path / "f.png")
    rc, _, err = (probar_imagen.main(["--imagen", str(tmp_path / "f.png"), "--modelo-calidad", destino,
                                      "--modelo-morfologia", destino, "--sin-grafico"]),) + tuple(
        (lambda o: (o.out, o.err))(capsys.readouterr()))
    assert rc == COD_ENTRADA and "VECTORES" in err


def test_legado_declarado_no_se_acepta_para_un_csv_v2(v2, capsys):
    destino = str(v2["dir"] / "x.json")
    rc, _, err = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino,
                         "--preprocesamiento", "legado"], capsys)
    assert rc == COD_ENTRADA and "sin_oscurecimiento" in err and not os.path.exists(destino)


def test_legado_declarado_si_se_acepta_para_el_dataset_entregado(tmp_path, capsys):
    destino = str(tmp_path / "x.json")
    rc, _, _ = correr(["--csv", os.path.join(RAIZ, "datasets", "dataset_calidad_real.csv"), "--rho", "0.7",
                       "--guardar-modelo", destino, "--preprocesamiento", "legado"], capsys)
    m = cargar_modelo(destino)
    assert rc == COD_OK and m.tipo == "fotografias" and m.preprocesamiento["version"] == "v1"


def test_dataset_entregado_sin_declaracion_no_se_supone_v1(tmp_path, capsys):
    destino = str(tmp_path / "x.json")
    rc, out, _ = correr(["--csv", os.path.join(RAIZ, "datasets", "dataset_calidad_real.csv"), "--rho", "0.7",
                         "--guardar-modelo", destino], capsys)
    assert rc == COD_OK and cargar_modelo(destino).tipo == "vectores"


@pytest.mark.parametrize("mutar,texto", [
    (lambda f: [r for r in f if r[1] != "sin_oscurecimiento"], "no coinciden"),                       # falta una variable
    (lambda f: [r if r[1] != "es_grande" else ["variable", "es_grande", "inventada", ">=", "1.0"] for r in f],
     "medida continua desconocida"),
    (lambda f: [r if r[1] != "es_grande" else ["variable", "es_grande", "area", "~", "1.0"] for r in f],
     "operador desconocido"),
    (lambda f: [r if r[1] != "es_grande" else ["variable", "es_grande", "area", ">=", "mucho"] for r in f],
     "umbral no numerico"),
    (lambda f: f + [["variable", "es_grande", "area", ">=", "1.0"]], "repetida"),
    (lambda f: [r if r[1] != "etapa" else ["global", "etapa", "", "", "morfologia"] for r in f], "no es el orden esperado"),
])
def test_preprocesamiento_incompleto_o_incompatible_se_rechaza_sin_escribir_el_modelo(v2, capsys, mutar, texto):
    ruta = v2["dir"] / "preprocesamiento_calidad.csv"
    filas = list(csv.reader(open(ruta, newline="")))
    cab, cuerpo = filas[0], filas[1:]
    nuevo = mutar(cuerpo)
    with open(ruta, "w", newline="") as f:
        csv.writer(f).writerows([cab] + nuevo)
    destino = str(v2["dir"] / "x.json")
    rc, _, err = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino], capsys)
    assert rc == COD_ENTRADA and texto in err and not os.path.exists(destino)


def test_orden_de_variables_distinto_se_rechaza(v2, capsys):
    ruta = v2["dir"] / "preprocesamiento_calidad.csv"
    filas = list(csv.reader(open(ruta, newline="")))
    i, j = [k for k, r in enumerate(filas) if r[1] in ("es_grande", "es_pesada")]
    filas[i], filas[j] = filas[j], filas[i]
    with open(ruta, "w", newline="") as f:
        csv.writer(f).writerows(filas)
    destino = str(v2["dir"] / "x.json")
    rc, _, err = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", destino], capsys)
    assert rc == COD_ENTRADA and "no coinciden" in err and not os.path.exists(destino)


def test_preprocesamiento_de_otra_etapa_se_rechaza(v2, tmp_path, capsys):
    pm = X.configuracion("morfologia", {c: 1.0 for c in ("es_esferica", "cascara_rugosa", "tiene_tallo", "densidad_alta",
                                                         "pigmentacion_oscura", "color_uniforme")},
                         segmentador="v1", tallo="convexidad", tono="circular")
    X.escribir_preprocesamiento_csv(pm, str(tmp_path / "morf.csv"))
    rc, _, err = correr(["--csv", v2["csv"], "--rho", "0.6", "--guardar-modelo", str(tmp_path / "x.json"),
                         "--preprocesamiento", str(tmp_path / "morf.csv")], capsys)
    assert rc == COD_ENTRADA and "no coinciden" in err


def test_csv_de_preprocesamiento_ida_y_vuelta_es_exacta(v2, tmp_path):
    pre = v2["pre"]
    leido = X.leer_preprocesamiento_csv(str(v2["dir"] / "preprocesamiento_calidad.csv"), pre["cols_features"])
    for k in ("version", "etapa", "segmentador", "tam_canvas", "cols_features", "reglas", "umbrales", "origen_umbrales"):
        assert leido[k] == pre[k]
    assert leido["opciones"]["extras"] == list(EXTRAS_CAL) and leido["opciones"]["color"] == "mascara"


def test_guardar_exige_coincidencia_entre_preprocesamiento_y_variables_del_modelo(v2, tmp_path):
    ds = val.leer_dataset(v2["csv"])
    red, meta = CarGross.entrenar_desde_dataset(ds, 0.6)
    pre = dict(v2["pre"])
    pre["cols_features"] = list(reversed(pre["cols_features"]))
    with pytest.raises(val.ErrorEntrada):
        red.guardar(str(tmp_path / "x.json"), meta, preprocesamiento=pre)
    assert not (tmp_path / "x.json").exists()


def test_medidas_conocidas_coinciden_con_las_que_produce_medir_foto():
    img = np.full((240, 320, 3), (215, 225, 230), np.uint8)
    cv2.circle(img, (160, 125), 70, (40, 60, 200), -1)
    assert set(X.medir_foto(img, segmentador="v1").crudos) == set(val.MEDIDAS_CONOCIDAS)


def test_toda_variable_de_los_esquemas_tiene_regla_construible():
    for etapa, esq in val.ESQUEMAS.items():
        for col in esq.features_extra:
            crudo, op = X._regla_extra(etapa, col, "mascara")
            assert crudo in val.MEDIDAS_CONOCIDAS and op in (">=", "<=")
            for color in X.COLOR_SUFIJOS:
                assert X._regla_extra(etapa, col, color)[0] in val.MEDIDAS_CONOCIDAS
