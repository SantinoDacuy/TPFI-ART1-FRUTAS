"""Entregables de la segunda ronda: la variante principal conserva las definiciones comprometidas, no usa el
informe para entrenar, el preprocesamiento viaja con el modelo y todo se reproduce desde los CSV."""
import csv
import json
import os

import pytest

import CarGross
import extraccion as X
import validacion as val
from CarGross import cargar_modelo
from conftest import RAIZ
from validacion import COD_OK

DS = os.path.join(RAIZ, "datasets_v2")
MD = os.path.join(RAIZ, "modelos_v2")
VARIANTES = ("principal", "alternativa_orden_canonico", "experimental_medianas")
hay = os.path.isfile(os.path.join(MD, "principal", "art1_morfologia.json"))
pytestmark = pytest.mark.skipif(not hay, reason="faltan modelos_v2/principal (scripts/final_ronda2.py)")


def ids(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return [r["fruta_id"] for r in csv.DictReader(f)]


@pytest.mark.parametrize("var", VARIANTES)
@pytest.mark.parametrize("etapa", ("morfologia", "calidad"))
def test_modelo_formato_2_de_fotografias_con_preprocesamiento_del_csv(var, etapa):
    m = cargar_modelo(os.path.join(MD, var, f"art1_{etapa}.json"))
    assert m.formato == 2 and m.tipo == "fotografias" and not m.legado and not m.advertencias
    pre_csv = X.leer_preprocesamiento_csv(os.path.join(DS, var, f"preprocesamiento_{etapa}.csv"), m.cols_features)
    for k in ("version", "etapa", "segmentador", "tam_canvas", "cols_features", "reglas", "umbrales"):
        assert pre_csv[k] == m.preprocesamiento[k]
    assert m.meta["preprocesamiento_origen"].endswith(f"{var}/preprocesamiento_{etapa}.csv")
    assert m.meta["procedencia"]["semilla_informe"] is not None


@pytest.mark.parametrize("var", VARIANTES)
@pytest.mark.parametrize("etapa", ("morfologia", "calidad"))
def test_el_informe_no_se_usa_para_entrenar(var, etapa):
    entr = set(ids(os.path.join(DS, var, f"{etapa}_entrenamiento.csv")))
    inf = set(ids(os.path.join(DS, var, f"{etapa}_informe.csv")))
    assert entr and inf and not (entr & inf)
    with open(os.path.join(DS, "manifiesto_imagenes.csv"), newline="", encoding="utf-8") as f:
        bloque = {r["foto_id"]: r["bloque"] for r in csv.DictReader(f)}
    assert all(bloque[i] == "desarrollo" for i in entr) and all(bloque[i] == "informe" for i in inf)


@pytest.mark.parametrize("var", VARIANTES)
@pytest.mark.parametrize("etapa", ("morfologia", "calidad"))
def test_el_modelo_se_reproduce_desde_el_csv_con_la_cli(var, etapa, tmp_path, capsys):
    m = cargar_modelo(os.path.join(MD, var, f"art1_{etapa}.json"))
    destino = str(tmp_path / "m.json")
    ent = m.meta["entrenamiento"]
    rc = CarGross.main(["--csv", os.path.join(DS, var, f"{etapa}_entrenamiento.csv"), "--rho", str(m.red.rho),
                        "--epocas", str(ent["epocas_pedidas"]), "--desempate", ent["desempate"],
                        "--guardar-modelo", destino])
    capsys.readouterr()
    assert rc == COD_OK
    n = cargar_modelo(destino)
    assert n.red.top_down == m.red.top_down
    assert json.loads(json.dumps(n.meta["mapeo_clusters"])) == json.loads(json.dumps(m.meta["mapeo_clusters"]))
    assert n.preprocesamiento == m.preprocesamiento and n.tipo == "fotografias"


def test_la_principal_usa_los_parametros_de_config_sin_pasarlos(tmp_path, capsys):
    import configuracion
    p = configuracion.parametros_etapa("morfologia")
    m = cargar_modelo(os.path.join(MD, "principal", "art1_morfologia.json"))
    assert m.red.rho == p["rho"] and m.meta["entrenamiento"]["epocas_pedidas"] == p["epocas"]
    destino = str(tmp_path / "m.json")
    rc = CarGross.main(["--csv", os.path.join(DS, "principal", "morfologia_entrenamiento.csv"), "--guardar-modelo", destino])
    capsys.readouterr()
    assert rc == COD_OK and cargar_modelo(destino).red.top_down == m.red.top_down


def test_la_principal_conserva_las_definiciones_comprometidas():
    mm = cargar_modelo(os.path.join(MD, "principal", "art1_morfologia.json"))
    mc = cargar_modelo(os.path.join(MD, "principal", "art1_calidad.json"))
    esq_m, esq_c = val.ESQUEMAS["morfologia"], val.ESQUEMAS["calidad"]
    assert mm.cols_features[:6] == list(esq_m.features_base) and mc.cols_features[:5] == list(esq_c.features_base)
    um = mm.preprocesamiento["umbrales"]
    assert {c: um[c] for c in esq_m.features_base} == X.CORTES_ORIGINALES_MORFOLOGIA | {"tiene_tallo": 0.06, "color_uniforme": 12.0}
    assert mm.preprocesamiento["reglas"]["tiene_tallo"]["crudo"] == "defecto_convexidad"      # mismo criterio, OpenCV 5
    assert mm.preprocesamiento["segmentador"] == "v1" and mm.preprocesamiento["opciones"]["tono"] == "circular"
    # los descriptores adicionales son cromaticos y estan documentados en el esquema
    assert set(mm.cols_features[6:]) <= set(esq_m.features_extra) and set(mc.cols_features[5:]) <= set(esq_c.features_extra)
    assert all(mm.preprocesamiento["reglas"][c]["crudo"] in val.MEDIDAS_CONOCIDAS for c in mm.cols_features)


def test_la_principal_cumple_el_criterio_de_exito_de_categorias_del_primer_avance():
    for etapa in ("morfologia", "calidad"):
        assert len(cargar_modelo(os.path.join(MD, "principal", f"art1_{etapa}.json")).red.top_down) < 15


def test_la_variante_de_medianas_esta_separada_y_marcada_experimental():
    m = cargar_modelo(os.path.join(MD, "experimental_medianas", "art1_morfologia.json"))
    assert m.meta["comprometido_con_el_primer_avance"] is False
    assert m.preprocesamiento["umbrales"]["es_esferica"] != 0.8              # cambia el significado de la variable
    assert cargar_modelo(os.path.join(MD, "principal", "art1_morfologia.json")).meta["comprometido_con_el_primer_avance"] is True


def test_predicciones_del_modelo_guardado_coinciden_con_las_del_informe_publicado(tmp_path, capsys):
    ruta_pub = os.path.join(RAIZ, "resultados", "ronda2", "informe_fotos_principal.csv")
    if not os.path.isfile(ruta_pub):
        pytest.skip("falta resultados/ronda2/informe_fotos_principal.csv")
    sal = str(tmp_path / "p.csv")
    rc = CarGross.main(["--csv", os.path.join(DS, "principal", "morfologia_informe.csv"), "--inferir-con",
                        os.path.join(MD, "principal", "art1_morfologia.json"), "--salida-predicciones", sal])
    capsys.readouterr()
    assert rc == COD_OK
    pred = {r["fruta_id"]: r["etiqueta_predicha"] for r in csv.DictReader(open(sal, encoding="utf-8"))}
    pub = {r["foto_id"]: r["morf_etiqueta"] for r in csv.DictReader(open(ruta_pub, encoding="utf-8"))}
    assert pred == pub


def test_los_datasets_de_los_modelos_estan_en_csv_validos():
    for var in VARIANTES:
        for etapa in ("morfologia", "calidad"):
            for part in ("entrenamiento", "informe"):
                ds = val.leer_dataset(os.path.join(DS, var, f"{etapa}_{part}.csv"))
                assert ds.esquema == etapa and ds.etiquetas
