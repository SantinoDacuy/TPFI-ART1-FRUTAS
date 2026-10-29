"""Validacion de entradas: archivos vacios, encabezados, valores ausentes o no
binarios, identificadores duplicados, dimensiones y modo generico."""
import os

import pytest

from conftest import RAIZ

import validacion as val
from validacion import ErrorEntrada

ENC_M = "fruta_id,es_esferica,cascara_rugosa,tiene_tallo,densidad_alta,pigmentacion_oscura,color_uniforme,tipo_real"
ENC_C = "fruta_id,es_grande,es_pesada,textura_firme,sin_defectos_superficie,color_homogeneo,calidad_real"


def csv_en(tmp_path, texto, nombre="d.csv"):
    p = tmp_path / nombre
    p.write_text(texto, encoding="utf-8")
    return str(p)


def problemas(ruta, **kw):
    with pytest.raises(ErrorEntrada) as e:
        val.leer_dataset(ruta, **kw)
    return " | ".join(e.value.problemas)


def test_dataset_valido_conserva_columnas_y_orden(tmp_path):
    ds = val.leer_dataset(csv_en(tmp_path, f"{ENC_M}\nA1,1,0,0,1,0,1,carozo\nA2,0,1,0,1,1,0,citrico\n"))
    assert ds.esquema == "morfologia"
    assert ds.cols_features == list(val.ESQUEMA_MORFOLOGIA.features_base)
    assert ds.etiquetas == ["carozo", "citrico"] and ds.ids == ["A1", "A2"]


def test_datasets_entregados_son_validos():
    for ruta, esq, n in ((os.path.join(RAIZ, "baseline_original", "datasets", "dataset_morfologia.csv"), "morfologia", 6),
                         (os.path.join(RAIZ, "baseline_original", "datasets", "dataset_calidad_real.csv"), "calidad", 5)):
        ds = val.leer_dataset(ruta)
        assert ds.esquema == esq and len(ds) == 180 and len(ds.cols_features) == n


def test_archivo_inexistente(tmp_path):
    assert "no se encontro" in problemas(str(tmp_path / "nada.csv"))


def test_archivo_vacio_y_solo_espacios(tmp_path):
    assert "vacio" in problemas(csv_en(tmp_path, ""))
    assert "no contiene" in problemas(csv_en(tmp_path, "\n\n", "e2.csv"))


def test_solo_encabezado(tmp_path):
    assert "ninguna fila" in problemas(csv_en(tmp_path, ENC_M + "\n"))


def test_encabezado_incorrecto_y_columnas_faltantes(tmp_path):
    assert "ningun esquema conocido" in problemas(csv_en(tmp_path, "a,b,c\n1,0,1\n"))
    t = problemas(csv_en(tmp_path, "fruta_id,es_esferica,tipo_real\nA,1,carozo\n", "f.csv"))
    assert "faltan caracteristicas obligatorias" in t


def test_columna_repetida_y_sin_id(tmp_path):
    t = problemas(csv_en(tmp_path, ENC_M.replace("tiene_tallo", "es_esferica") + "\nA,1,0,0,1,0,1,carozo\n"))
    assert "repetidas" in t
    t = problemas(csv_en(tmp_path, ENC_M.replace("fruta_id", "ident") + "\nA,1,0,0,1,0,1,carozo\n", "g.csv"))
    assert "fruta_id" in t


def test_valor_ausente_no_se_convierte_en_metadato(tmp_path):
    t = problemas(csv_en(tmp_path, f"{ENC_M}\nA,1,,0,1,0,1,carozo\n"))
    assert "linea 2" in t and "'cascara_rugosa'" in t and "ausente" in t


@pytest.mark.parametrize("valor", ["2", "0.5", "si", "-1", "01", "1.0"])
def test_valor_no_binario_es_error_con_fila_y_columna(tmp_path, valor):
    t = problemas(csv_en(tmp_path, f"{ENC_C}\nP1,1,1,1,1,1,premium\nP2,1,{valor},1,1,1,premium\n"))
    assert "linea 3" in t and "'es_pesada'" in t and "no binario" in t


def test_identificadores_duplicados_y_vacios(tmp_path):
    t = problemas(csv_en(tmp_path, f"{ENC_C}\nP1,1,1,1,1,1,premium\nP1,0,1,1,1,1,premium\n,0,0,0,0,0,descarte\n"))
    assert "duplicado 'P1'" in t and "identificador vacio" in t


def test_etiqueta_invalida_o_ausente(tmp_path):
    t = problemas(csv_en(tmp_path, f"{ENC_C}\nP1,1,1,1,1,1,excelente\nP2,1,1,1,1,1,\n"))
    assert "'excelente'" in t and "etiqueta ausente" in t


def test_fila_con_campos_de_mas_o_de_menos(tmp_path):
    t = problemas(csv_en(tmp_path, f"{ENC_C}\nP1,1,1,1,1,premium\nP2,1,1,1,1,1,1,premium\n"))
    assert t.count("campos") == 2


def test_columna_desconocida_no_se_adopta_en_silencio(tmp_path):
    t = problemas(csv_en(tmp_path, ENC_C + ",peso\nP1,1,1,1,1,1,premium,12\n"))
    assert "'peso'" in t and "generico" in t


def test_reporta_todos_los_problemas_juntos(tmp_path):
    t = problemas(csv_en(tmp_path, f"{ENC_C}\nP1,2,1,1,1,1,premium\nP2,1,x,1,1,1,premium\nP3,1,1,1,1,,premium\n"))
    assert t.count("linea") == 3


def test_utf8_con_bom(tmp_path):
    p = tmp_path / "b.csv"
    p.write_bytes(b"\xef\xbb\xbf" + f"{ENC_C}\nP1,1,1,1,1,1,premium\n".encode())
    assert val.leer_dataset(str(p)).ids == ["P1"]


def test_vector_nulo_genera_advertencia(tmp_path):
    ds = val.leer_dataset(csv_en(tmp_path, f"{ENC_C}\nP1,0,0,0,0,0,descarte\nP2,1,1,1,1,1,premium\n"))
    assert any("vector nulo" in a for a in ds.advertencias)


# --- modo generico ----------------------------------------------------------------
def test_generico_requiere_etiqueta_declarada_si_hay_columnas_real(tmp_path):
    ruta = csv_en(tmp_path, "id,x,y,clase_real\n1,1,0,a\n2,0,1,b\n")
    assert "etiquetas de referencia" in problemas(ruta, esquema="generico", columna_id="id")
    ds = val.leer_dataset(ruta, esquema="generico", columna_id="id", col_etiqueta="clase_real")
    assert ds.cols_features == ["x", "y"] and ds.etiquetas == ["a", "b"]


def test_generico_columna_no_binaria_no_pasa_a_metadato(tmp_path):
    ruta = csv_en(tmp_path, "id,x,y,clase\n1,1,0,a\n2,0,2,b\n")
    t = problemas(ruta, esquema="generico", columna_id="id", col_etiqueta="clase")
    assert "'y'" in t and "no binario" in t


def test_generico_metadato_debe_declararse(tmp_path):
    ruta = csv_en(tmp_path, "id,x,y,nota\n1,1,0,hola\n")
    assert "'nota'" in problemas(ruta, esquema="generico", columna_id="id")
    ds = val.leer_dataset(ruta, esquema="generico", columna_id="id", cols_metadata=["nota"])
    assert ds.cols_features == ["x", "y"] and ds.metadata[0] == {"nota": "hola"}


def test_generico_etiqueta_inexistente(tmp_path):
    assert "no existe" in problemas(csv_en(tmp_path, "id,x\n1,1\n"), esquema="generico",
                                    columna_id="id", col_etiqueta="zzz")


def test_esquema_desconocido(tmp_path):
    assert "esquema desconocido" in problemas(csv_en(tmp_path, f"{ENC_C}\nP,1,1,1,1,1,premium\n"), esquema="otro")


# --- compatibilidad con el modelo -------------------------------------------------
def test_compatibilidad_con_modelo(tmp_path):
    ds = val.leer_dataset(csv_en(tmp_path, f"{ENC_C}\nP1,1,0,1,0,1,premium\n"))
    cols = ["color_homogeneo", "es_grande", "es_pesada", "textura_firme", "sin_defectos_superficie"]
    assert val.reordenar_a_modelo(ds, cols) == [[1, 1, 0, 1, 0]]  # alinea por nombre
    with pytest.raises(ErrorEntrada, match="dimensiones incompatibles"):
        val.validar_compatibilidad(ds, cols + ["extra"])
    with pytest.raises(ErrorEntrada, match="no usa"):
        val.validar_compatibilidad(ds, cols[:-1])
