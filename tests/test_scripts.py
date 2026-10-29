"""Scripts de resultados y demo: fallan con codigo y mensaje utiles cuando faltan insumos."""
import os
import subprocess
import sys

import pytest

import configuracion
import demo
from conftest import RAIZ
from validacion import COD_ENTRADA


def correr(script, *args):
    return subprocess.run([sys.executable, os.path.join(RAIZ, "scripts", script), *args],
                          capture_output=True, text=True, cwd=RAIZ)


def test_experimentos_sin_mediciones_devuelve_3_y_explica(tmp_path):
    p = correr("experimentos.py", "ablacion", "--mediciones", str(tmp_path))
    assert p.returncode == COD_ENTRADA and "extraer_mediciones" in p.stderr


def test_experimentos_final_sin_ablacion_devuelve_3(tmp_path):
    p = correr("experimentos.py", "final", "--salida", str(tmp_path),
               "--mediciones", os.path.join(RAIZ, "resultados", "mediciones"))
    assert p.returncode == COD_ENTRADA and "ablacion" in p.stderr


def test_extraer_mediciones_sin_imagenes_devuelve_3(tmp_path):
    p = correr("extraer_mediciones.py", "--imagenes", str(tmp_path), "--salida", str(tmp_path / "m"))
    assert p.returncode == COD_ENTRADA and "no existe la carpeta" in p.stderr


def test_demo_no_genera_ni_borra_datasets(monkeypatch, capsys):
    """La demo con --sin-fotos solo ejecuta subprocesos de lectura; los datasets entregados no cambian."""
    h = {n: open(os.path.join(RAIZ, "datasets", n), "rb").read() for n in os.listdir(os.path.join(RAIZ, "datasets"))
         if n.endswith(".csv")}
    rc = demo.main(["--sin-fotos", "--rho-morf", "0.7", "--rho-cal", "0.7"])
    assert rc == 0
    assert {n: open(os.path.join(RAIZ, "datasets", n), "rb").read() for n in h} == h


def test_demo_propaga_corridas_infructuosas(monkeypatch, capsys):
    """Si una corrida falla (aqui: archivo inexistente), la demo termina distinto de cero y lo informa."""
    monkeypatch.setattr(demo, "CSV_MORF", os.path.join(RAIZ, "datasets", "no_existe.csv"))
    rc = demo.main(["--sin-fotos"])
    assert rc == COD_ENTRADA or rc == 1


def test_configuracion_lee_y_valida_el_csv(tmp_path):
    p = configuracion.parametros_etapa("calidad")
    assert 0 < p["rho"] <= 1 and p["L"] > 1 and p["politica_nulo"] in ("categoria_propia", "rechazar")
    malo = tmp_path / "p.csv"
    malo.write_text("etapa,rho,L,max_categorias,epocas,politica_nulo,desempate,fuente\nraro,0.5,2,50,1,x,y,z\n")
    with pytest.raises(ValueError, match="etapa desconocida"):
        configuracion.cargar_parametros(str(malo))
    malo.write_text("etapa,rho,L,max_categorias,epocas,politica_nulo,desempate,fuente\nmorfologia,abc,2,50,1,x,y,z\n")
    with pytest.raises(ValueError, match="invalida"):
        configuracion.cargar_parametros(str(malo))
    malo.write_text("etapa,rho,L,max_categorias,epocas,politica_nulo,desempate,fuente\nmorfologia,0.5,2,50,1,x,y,z\n")
    with pytest.raises(ValueError, match="faltan las etapas"):
        configuracion.cargar_parametros(str(malo))
    with pytest.raises(ValueError, match="no se encontro"):
        configuracion.cargar_parametros(str(tmp_path / "no.csv"))
