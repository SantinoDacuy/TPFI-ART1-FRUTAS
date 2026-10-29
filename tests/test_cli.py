"""CLI de CarGross.py: codigos de salida, diagnosticos, ayuda, modo generico y
guardado de modelos."""
import os
import subprocess
import sys

import pytest

import CarGross
from CarGross import cargar_modelo
from conftest import RAIZ
from validacion import COD_ENTRADA, COD_INFRUCTUOSA, COD_OK, COD_USO

CSV_MORF = os.path.join(RAIZ, "datasets", "dataset_morfologia.csv")
CSV_CAL = os.path.join(RAIZ, "datasets", "dataset_calidad_real.csv")


def correr(argv, capsys):
    rc = CarGross.main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_corrida_exitosa_devuelve_cero_y_reporta(capsys):
    rc, out, _ = correr(["--csv", CSV_MORF, "--rho", "0.7", "--epocas", "1"], capsys)
    assert rc == COD_OK
    assert "Corrida exitosa - rho=0.7 - 14 clusters" in out and "76.11%" in out
    assert "no es la exactitud de prueba" in out


def test_ayuda_muestra_opciones_y_sale_con_cero(capsys):
    for flag in ("--help", "--ayuda"):
        with pytest.raises(SystemExit) as e:
            CarGross.main([flag])
        assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--esquema" in out and "--guardar-modelo" in out and "--sobrescribir" in out


def test_argumentos_invalidos_salen_con_2(capsys):
    with pytest.raises(SystemExit) as e:
        CarGross.main(["--rho", "0.7"])           # falta --csv
    assert e.value.code == COD_USO
    with pytest.raises(SystemExit) as e:
        CarGross.main(["--csv", CSV_MORF, "--orden", "raro"])
    assert e.value.code == COD_USO


def test_archivo_inexistente_codigo_3_con_diagnostico_y_ayuda(capsys, tmp_path):
    rc, out, err = correr(["--csv", str(tmp_path / "no.csv"), "--rho", "0.7"], capsys)
    assert rc == COD_ENTRADA
    assert "no se encontro el archivo" in err and "--help" in err


def test_csv_invalido_codigo_3_y_localiza_el_error(capsys, tmp_path):
    p = tmp_path / "malo.csv"
    p.write_text("fruta_id,es_grande,es_pesada,textura_firme,sin_defectos_superficie,color_homogeneo,calidad_real\n"
                 "A,1,1,1,1,1,premium\nB,1,2,1,1,1,premium\n")
    rc, out, err = correr(["--csv", str(p), "--rho", "0.7"], capsys)
    assert rc == COD_ENTRADA and "linea 3" in err and "'es_pesada'" in err and "no binario" in err


@pytest.mark.parametrize("rho", ["1.5", "0", "-0.2"])
def test_rho_invalido_codigo_2(capsys, rho):
    rc, _, err = correr(["--csv", CSV_MORF, "--rho", rho], capsys)
    assert rc == COD_USO and "configuracion invalida" in err


def test_corrida_infructuosa_devuelve_distinto_de_cero(capsys):
    rc, out, err = correr(["--csv", CSV_MORF, "--rho", "0.95", "--max-categorias", "5", "--epocas", "1"], capsys)
    assert rc == COD_INFRUCTUOSA
    assert "CORRIDA INFRUCTUOSA" in out and "corrida infructuosa" in err


def test_codigo_de_salida_real_del_proceso():
    base = [sys.executable, os.path.join(RAIZ, "CarGross.py")]
    p = subprocess.run(base + ["--csv", CSV_MORF, "--rho", "0.95", "--max-categorias", "5", "--epocas", "1"],
                       capture_output=True, text=True)
    assert p.returncode == COD_INFRUCTUOSA
    p = subprocess.run(base + ["--csv", "no_existe.csv", "--rho", "0.5"], capture_output=True, text=True)
    assert p.returncode == COD_ENTRADA
    p = subprocess.run(base + ["--csv", CSV_CAL, "--rho", "0.7", "--epocas", "1"], capture_output=True, text=True)
    assert p.returncode == COD_OK


def test_rho_desde_configuracion_si_se_omite(capsys):
    rc, out, _ = correr(["--csv", CSV_CAL], capsys)
    assert rc == COD_OK and "tomado de config/parametros_etapas.csv" in out


def test_guardar_modelo_valido_y_sin_pisar(capsys, tmp_path):
    ruta = str(tmp_path / "m.json")
    rc, out, _ = correr(["--csv", CSV_CAL, "--rho", "0.7", "--guardar-modelo", ruta], capsys)
    assert rc == COD_OK and "formato 2" in out
    m = cargar_modelo(ruta)
    assert m.formato == 2 and m.meta["entrenamiento"]["modo_mapeo"] == "prototipos_finales"
    antes = open(ruta, "rb").read()
    rc, _, err = correr(["--csv", CSV_CAL, "--rho", "0.8", "--guardar-modelo", ruta], capsys)
    assert rc == COD_USO and "ya existe" in err and open(ruta, "rb").read() == antes
    rc, _, _ = correr(["--csv", CSV_CAL, "--rho", "0.8", "--guardar-modelo", ruta, "--sobrescribir"], capsys)
    assert rc == COD_OK and cargar_modelo(ruta).red.rho == 0.8


def test_modelo_guardado_por_cli_reproduce_el_entrenamiento(capsys, tmp_path):
    ruta = str(tmp_path / "m.json")
    correr(["--csv", CSV_MORF, "--rho", "0.8", "--epocas", "3", "--guardar-modelo", ruta], capsys)
    rc, _, _ = correr(["--csv", CSV_MORF, "--rho", "0.8", "--epocas", "3", "--guardar-modelo",
                       str(tmp_path / "m2.json")], capsys)
    a, b = cargar_modelo(ruta), cargar_modelo(str(tmp_path / "m2.json"))
    assert a.red.top_down == b.red.top_down and a.meta["mapeo_clusters"] == b.meta["mapeo_clusters"]


def test_modo_generico_explicito(capsys, tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("id,x,y,z,clase\n1,1,0,1,a\n2,1,1,1,a\n3,0,0,1,b\n4,0,1,0,b\n")
    rc, _, err = correr(["--csv", str(p), "--id-col", "id", "--rho", "0.5"], capsys)
    assert rc == COD_ENTRADA and "esquema" in err                     # no se adivina en silencio
    rc, _, err = correr(["--csv", str(p), "--id-col", "id", "--esquema", "generico", "--col-etiqueta", "clase"], capsys)
    assert rc == COD_USO and "--rho" in err                           # sin esquema conocido no hay rho por defecto
    rc, out, _ = correr(["--csv", str(p), "--id-col", "id", "--esquema", "generico", "--col-etiqueta", "clase",
                         "--rho", "0.5"], capsys)
    assert rc == COD_OK and "clusters formados" in out


def test_politica_nulo_rechazar_con_datos_nulos_falla_con_3(capsys):
    rc, out, err = correr(["--csv", CSV_CAL, "--rho", "0.7", "--politica-nulo", "rechazar"], capsys)
    assert rc == COD_ENTRADA and "vectores nulos" in err


def test_aviso_de_vectores_nulos_en_el_dataset_de_calidad(capsys):
    rc, out, _ = correr(["--csv", CSV_CAL, "--rho", "0.7", "--epocas", "1"], capsys)
    assert "16 fila(s) con vector nulo" in out


def test_error_interno_devuelve_4(capsys, monkeypatch):
    monkeypatch.setattr(CarGross, "ejecutar", lambda *a, **k: 1 / 0)
    rc, _, err = correr(["--csv", CSV_CAL, "--rho", "0.7"], capsys)
    assert rc == 4 and "--depurar" in err


def test_migrar_modelo_por_cli(capsys, tmp_path):
    origen = os.path.join(RAIZ, "modelos", "art1_calidad.json")
    sal = str(tmp_path / "mig.json")
    rc, out, _ = correr(["--csv", CSV_CAL, "--migrar-modelo", origen, "--guardar-modelo", sal], capsys)
    assert rc == COD_OK and cargar_modelo(sal).formato == 2
    rc, _, err = correr(["--csv", CSV_CAL, "--migrar-modelo", origen], capsys)
    assert rc == COD_USO and "--guardar-modelo" in err


# --- inferencia sobre CSV con un modelo guardado ---------------------------------------------------
def test_inferir_con_modelo_no_modifica_pesos_ni_archivo(capsys, tmp_path):
    modelo = os.path.join(RAIZ, "modelos", "art1_calidad.json")
    antes = open(modelo, "rb").read()
    sal = str(tmp_path / "pred.csv")
    rc, out, err = correr(["--csv", CSV_CAL, "--inferir-con", modelo, "--salida-predicciones", sal], capsys)
    assert rc == COD_OK and "pesos sin cambios" in out and "REINFERENCIA" in out and "Matriz de confusion" in out
    assert "formato 1" in out                                   # advertencia del modo legado
    assert open(modelo, "rb").read() == antes
    import csv
    filas = list(csv.DictReader(open(sal, encoding="utf-8")))
    assert len(filas) == 180 and set(filas[0]) >= {"fruta_id", "categoria", "etiqueta_predicha", "coincidencia"}


def test_inferir_con_dimensiones_incompatibles_codigo_3(capsys):
    rc, _, err = correr(["--csv", CSV_MORF, "--inferir-con", os.path.join(RAIZ, "modelos", "art1_calidad.json")], capsys)
    assert rc == COD_ENTRADA and "dimensiones incompatibles" in err and "faltan las variables del modelo" in err


def test_inferir_con_modelo_inexistente_o_corrupto(capsys, tmp_path):
    rc, _, err = correr(["--csv", CSV_CAL, "--inferir-con", str(tmp_path / "no.json")], capsys)
    assert rc == COD_ENTRADA and "no se encontro" in err
    roto = tmp_path / "roto.json"
    roto.write_text("{}")
    rc, _, err = correr(["--csv", CSV_CAL, "--inferir-con", str(roto)], capsys)
    assert rc == COD_ENTRADA and "falta la clave" in err


def test_inferir_alinea_variables_por_nombre_no_por_posicion(capsys, tmp_path):
    p = tmp_path / "reordenado.csv"
    p.write_text("fruta_id,color_homogeneo,es_grande,textura_firme,es_pesada,sin_defectos_superficie,calidad_real\n"
                 "A,1,1,1,1,1,premium\n")
    rc, out, _ = correr(["--csv", str(p), "--inferir-con", os.path.join(RAIZ, "modelos", "art1_calidad.json")], capsys)
    assert rc == COD_OK and "sobre 1 muestras" in out
