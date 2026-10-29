"""Guardado y carga de modelos: las predicciones no cambian; validacion al cargar;
compatibilidad con los modelos entregados (formato 1) y migracion."""
import itertools
import json
import os

import pytest

import extraccion as X
import validacion as val
from CarGross import FORMATO_MODELO, cargar_modelo, entrenar_desde_dataset, migrar_modelo_v1
from conftest import RAIZ
from validacion import ErrorModelo

ENC = "fruta_id,es_grande,es_pesada,textura_firme,sin_defectos_superficie,color_homogeneo,calidad_real"
FILAS = ["P1,1,1,1,1,1,premium", "P2,1,1,1,0,1,premium", "C1,0,1,0,1,1,comercial",
         "C2,0,0,0,1,1,comercial", "D1,0,0,0,0,0,descarte", "D2,0,0,1,0,0,descarte"]


@pytest.fixture
def ds(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text(ENC + "\n" + "\n".join(FILAS) + "\n")
    return val.leer_dataset(str(p))


def pre_calidad():
    return X.preprocesamiento_legado("calidad")


def modelo_guardado(ds, tmp_path, nombre="m.json", **kw):
    red, meta = entrenar_desde_dataset(ds, 0.6, **kw)
    ruta = str(tmp_path / nombre)
    red.guardar(ruta, meta, preprocesamiento=pre_calidad())
    return red, meta, ruta


def test_guardar_y_cargar_no_altera_predicciones(ds, tmp_path):
    red, meta, ruta = modelo_guardado(ds, tmp_path)
    m = cargar_modelo(ruta)
    assert m.formato == FORMATO_MODELO and not m.legado and not m.advertencias
    assert m.red.top_down == red.top_down and m.red.rho == red.rho and m.red.L == red.L
    for x in itertools.product([0, 1], repeat=5):
        a, b = red.evaluar(list(x)), m.red.evaluar(list(x))
        assert (a["ganador"], a["resuena"], a["ratio"]) == (b["ganador"], b["resuena"], b["ratio"])
        if a["resuena"]:
            assert m.categoria(a["ganador"])["etiqueta"] == meta["mapeo_clusters"][str(a["ganador"])]["etiqueta"]


def test_el_archivo_guarda_version_orden_y_umbrales_exactos(ds, tmp_path):
    _, _, ruta = modelo_guardado(ds, tmp_path)
    d = json.load(open(ruta, encoding="utf-8"))
    assert d["formato_version"] == 2
    assert d["metadata_extra"]["cols_features"] == list(ds.cols_features)
    pre = d["preprocesamiento"]
    assert pre["cols_features"] == list(ds.cols_features) and pre["tam_canvas"] == 200
    assert pre["umbrales"]["es_grande"] == 10320.5          # valor exacto, sin redondeo extra
    assert set(pre["reglas"]) == set(ds.cols_features)
    ent = d["metadata_extra"]["entrenamiento"]
    assert ent["csv_sha256"] and ent["orden"] == "original" and ent["modo_mapeo"] == "prototipos_finales"


def test_guardar_no_pisa_archivos_existentes(ds, tmp_path):
    red, meta, ruta = modelo_guardado(ds, tmp_path)
    antes = open(ruta, "rb").read()
    with pytest.raises(FileExistsError):
        red.guardar(ruta, meta, preprocesamiento=pre_calidad())
    assert open(ruta, "rb").read() == antes
    red.guardar(ruta, meta, preprocesamiento=pre_calidad(), sobrescribir=True)   # explicito: si


def test_sin_preprocesamiento_se_guarda_un_modelo_de_vectores(ds, tmp_path):
    red, meta = entrenar_desde_dataset(ds, 0.6)
    ruta = str(tmp_path / "v.json")
    red.guardar(ruta, meta)
    m = cargar_modelo(ruta)
    assert m.tipo == "vectores" and m.preprocesamiento is None
    assert json.load(open(ruta, encoding="utf-8"))["tipo_modelo"] == "vectores"


def test_preprocesamiento_incompleto_se_rechaza_antes_de_escribir(ds, tmp_path):
    red, meta = entrenar_desde_dataset(ds, 0.6)
    pre = pre_calidad()
    pre["cols_features"] = pre["cols_features"] + ["sin_oscurecimiento"]      # 6 variables, reglas de 5
    meta["cols_features"] = list(pre["cols_features"])
    ruta = tmp_path / "x.json"
    with pytest.raises(ErrorModelo, match="falta la regla de 'sin_oscurecimiento'"):
        red.guardar(str(ruta), meta, preprocesamiento=pre)
    assert not ruta.exists() and [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_no_quedan_temporales(ds, tmp_path):
    modelo_guardado(ds, tmp_path)
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


# --- validacion al cargar -------------------------------------------------------------
def cargar_modificado(tmp_path, ds, mutar):
    _, _, ruta = modelo_guardado(ds, tmp_path)
    d = json.load(open(ruta, encoding="utf-8"))
    mutar(d)
    p = tmp_path / "mod.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("mutar,texto", [
    (lambda d: d.pop("top_down"), "falta la clave 'top_down'"),
    (lambda d: d.update(rho=1.5), "rho fuera"),
    (lambda d: d.update(L=1.0), "L debe ser"),
    (lambda d: d.update(n_features=0), "n_features"),
    (lambda d: d["top_down"].append([1, 0]), "no es un vector binario de largo"),
    (lambda d: d["top_down"].__setitem__(0, [2] * 5), "no es un vector binario"),
    (lambda d: d.update(formato_version=9), "no soportado"),
    (lambda d: d.update(politica_nulo="otra"), "politica_nulo"),
    (lambda d: d["metadata_extra"].pop("cols_features"), "cols_features ausente"),
    (lambda d: d["metadata_extra"].update(cols_features=["a", "b"]), "no coincide"),
    (lambda d: d["metadata_extra"]["mapeo_clusters"].update({"99": {}}), "categoria inexistente"),
    (lambda d: d.update(preprocesamiento=None), "requiere la seccion"),
    (lambda d: d["preprocesamiento"].update(cols_features=["x"] * 5), "no coinciden"),
    (lambda d: d["preprocesamiento"].pop("umbrales"), "umbrales ausente"),
    (lambda d: d.update(max_categorias=1), "max_categorias"),
])
def test_modelo_invalido_se_rechaza_con_mensaje(tmp_path, ds, mutar, texto):
    ruta = cargar_modificado(tmp_path, ds, mutar)
    with pytest.raises(ErrorModelo) as e:
        cargar_modelo(ruta)
    assert texto in str(e.value)


def test_json_corrupto_y_archivo_ausente(tmp_path):
    p = tmp_path / "roto.json"
    p.write_text("{no es json")
    with pytest.raises(ErrorModelo, match="JSON invalido"):
        cargar_modelo(str(p))
    with pytest.raises(ErrorModelo, match="no se encontro"):
        cargar_modelo(str(tmp_path / "no.json"))


# --- compatibilidad con modelos entregados (formato 1) ------------------------------------
MODELOS_ORIG = os.path.join(RAIZ, "modelos")


@pytest.mark.parametrize("nombre,n,rho", [("art1_morfologia.json", 6, 0.85), ("art1_calidad.json", 5, 0.7)])
def test_modelos_entregados_cargan_en_modo_legado(nombre, n, rho):
    m = cargar_modelo(os.path.join(MODELOS_ORIG, nombre))
    assert m.legado and m.formato == 1 and m.preprocesamiento is None
    assert m.red.n == n and m.red.rho == rho and m.red.politica_nulo == "categoria_propia"
    assert m.advertencias and "formato 1" in m.advertencias[0]
    assert m.categoria(0)["etiqueta"] in ("citrico", "tropical", "carozo", "premium", "comercial", "descarte")
    pre = X.preprocesamiento_legado("morfologia" if n == 6 else "calidad")
    assert pre["cols_features"] == m.cols_features          # el orden legado coincide con el del archivo


def test_cargar_modelo_entregado_no_modifica_el_archivo():
    ruta = os.path.join(MODELOS_ORIG, "art1_calidad.json")
    antes = open(ruta, "rb").read()
    cargar_modelo(ruta)
    assert open(ruta, "rb").read() == antes


def test_migracion_v1_a_v2_conserva_pesos_y_no_pisa(tmp_path):
    origen = os.path.join(MODELOS_ORIG, "art1_calidad.json")
    csv_ = os.path.join(RAIZ, "datasets", "dataset_calidad_real.csv")
    salida = str(tmp_path / "migrado.json")
    migrar_modelo_v1(origen, csv_, salida, X.preprocesamiento_legado("calidad"))
    v1, v2 = cargar_modelo(origen), cargar_modelo(salida)
    assert v2.formato == 2 and v2.red.top_down == v1.red.top_down
    assert v2.meta["migracion"]["desde_formato"] == 1 and v2.meta["mapeo_clusters"]
    with pytest.raises(FileExistsError):
        migrar_modelo_v1(origen, csv_, salida, X.preprocesamiento_legado("calidad"))
    with pytest.raises(ErrorModelo, match="ya esta en formato 2"):
        migrar_modelo_v1(salida, csv_, str(tmp_path / "otro.json"), X.preprocesamiento_legado("calidad"))
