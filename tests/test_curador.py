"""Curador: seleccionar y eliminar son pasos separados; simulacion por defecto;
resultados en otra carpeta; nada se pisa ni se borra sin orden explicita."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import curar_dataset as cur
from conftest import RAIZ
from validacion import COD_ENTRADA, COD_OK, COD_USO


def banco_falso(tmp_path):
    """Banco con 4 fotos falsas y una seleccion donde 'b' y 'c' no las acepta ninguna etapa."""
    img = tmp_path / "imgs"
    for sub in ("Good Quality_Fruits/Apple_Good", "Mixed Qualit_Fruits/Apple"):
        os.makedirs(img / sub)
    nombres = {"a.jpg": "Good Quality_Fruits/Apple_Good", "b.jpg": "Good Quality_Fruits/Apple_Good",
               "c.jpg": "Mixed Qualit_Fruits/Apple", "d.jpg": "Mixed Qualit_Fruits/Apple"}
    for n, sub in nombres.items():
        (img / sub / n).write_bytes(b"x")
    sel = tmp_path / "seleccion.csv"
    with open(sel, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "etapa", "clase", "decision", "motivo"])
        w.writerows([
            ["Good Quality_Fruits/Apple_Good/a.jpg", "morfologia", "carozo", "aceptada", "OK"],
            ["Good Quality_Fruits/Apple_Good/a.jpg", "calidad", "premium", "rechazada", "x"],
            ["Good Quality_Fruits/Apple_Good/b.jpg", "morfologia", "carozo", "rechazada", "x"],
            ["Good Quality_Fruits/Apple_Good/b.jpg", "calidad", "premium", "excedente", "tope"],
            ["Mixed Qualit_Fruits/Apple/c.jpg", "calidad", "comercial", "rechazada", "x"],
            ["Mixed Qualit_Fruits/Apple/d.jpg", "calidad", "comercial", "aceptada", "OK"],
        ])
    return str(img), str(sel)


def test_imagenes_sin_uso_son_las_que_ninguna_etapa_acepta(tmp_path):
    _, sel = banco_falso(tmp_path)
    assert cur.imagenes_sin_uso(sel) == ["Good Quality_Fruits/Apple_Good/b.jpg", "Mixed Qualit_Fruits/Apple/c.jpg"]


def test_eliminar_es_simulacion_por_defecto(tmp_path, capsys):
    img, sel = banco_falso(tmp_path)
    antes = sorted(os.listdir(os.path.join(img, "Good Quality_Fruits", "Apple_Good")))
    rc = cur.main(["eliminar", "--seleccion", sel, "--imagenes", img])
    out = capsys.readouterr().out
    assert rc == COD_OK and "SIMULACION" in out and "2 imagen(es)" in out
    assert sorted(os.listdir(os.path.join(img, "Good Quality_Fruits", "Apple_Good"))) == antes


def test_ejecutar_mueve_y_no_borra(tmp_path, capsys):
    img, sel = banco_falso(tmp_path)
    dest = str(tmp_path / "descartadas")
    assert cur.main(["eliminar", "--seleccion", sel, "--imagenes", img, "--ejecutar", "--destino", dest]) == COD_OK
    assert not os.path.exists(os.path.join(img, "Good Quality_Fruits", "Apple_Good", "b.jpg"))
    assert os.path.isfile(os.path.join(dest, "Good Quality_Fruits", "Apple_Good", "b.jpg"))      # recuperable
    assert os.path.isfile(os.path.join(img, "Good Quality_Fruits", "Apple_Good", "a.jpg"))        # aceptadas: intactas


def test_borrado_definitivo_exige_confirmacion(tmp_path, capsys):
    img, sel = banco_falso(tmp_path)
    rc = cur.main(["eliminar", "--seleccion", sel, "--imagenes", img, "--ejecutar", "--borrar-definitivamente"])
    assert rc == COD_USO and os.path.isfile(os.path.join(img, "Mixed Qualit_Fruits", "Apple", "c.jpg"))
    rc = cur.main(["eliminar", "--seleccion", sel, "--imagenes", img, "--ejecutar", "--borrar-definitivamente",
                   "--confirmo-que-tengo-copia"])
    assert rc == COD_OK and not os.path.exists(os.path.join(img, "Mixed Qualit_Fruits", "Apple", "c.jpg"))


def test_mover_no_pisa_un_destino_existente(tmp_path, capsys):
    img, sel = banco_falso(tmp_path)
    dest = tmp_path / "descartadas" / "Good Quality_Fruits" / "Apple_Good"
    os.makedirs(dest)
    (dest / "b.jpg").write_bytes(b"previo")
    rc = cur.main(["eliminar", "--seleccion", sel, "--imagenes", img, "--ejecutar", "--destino", str(tmp_path / "descartadas")])
    assert rc == COD_USO and (dest / "b.jpg").read_bytes() == b"previo"
    assert os.path.isfile(os.path.join(img, "Good Quality_Fruits", "Apple_Good", "b.jpg"))


def test_eliminar_sin_seleccion_falla_con_3(tmp_path, capsys):
    rc = cur.main(["eliminar", "--seleccion", str(tmp_path / "nada.csv")])
    assert rc == COD_ENTRADA and "seleccionar" in capsys.readouterr().err


def test_seleccionar_no_pisa_salidas_existentes(tmp_path, capsys):
    salida = tmp_path / "salida"
    os.makedirs(salida)
    (salida / "seleccion_curacion.csv").write_text("previo")
    rc = cur.main(["seleccionar", "--imagenes", str(tmp_path), "--salida", str(salida)])
    assert rc == COD_USO and (salida / "seleccion_curacion.csv").read_text() == "previo"


def test_seleccionar_con_carpeta_inexistente_falla_con_3(tmp_path, capsys):
    rc = cur.main(["seleccionar", "--imagenes", str(tmp_path / "no"), "--salida", str(tmp_path / "s")])
    assert rc == COD_ENTRADA and "no existe la carpeta" in capsys.readouterr().err


def test_salida_por_defecto_no_es_datasets():
    assert os.path.basename(cur.SALIDA_POR_DEFECTO) == "salida_curacion"
    assert os.path.abspath(cur.SALIDA_POR_DEFECTO) != os.path.join(RAIZ, "datasets")


imagenes = os.path.join(RAIZ, "datasets", "processed_images_fruits")


@pytest.mark.skipif(not os.path.isdir(imagenes), reason="no estan las imagenes (carpeta fuera del repositorio)")
def test_la_seleccion_reproduce_los_csv_entregados_sin_tocar_nada(tmp_path):
    antes = {d: sorted(os.listdir(os.path.join(imagenes, *d.split("/")))) for d in
             ("Good Quality_Fruits/Apple_Good", "Mixed Qualit_Fruits/Apple", "Bad Quality_Fruits/Apple_Bad")}
    h = {n: open(os.path.join(RAIZ, "datasets", n), "rb").read() for n in
         ("dataset_morfologia.csv", "dataset_calidad_real.csv")}
    assert cur.main(["seleccionar", "--salida", str(tmp_path / "s")]) == COD_OK
    for n in h:
        a = open(tmp_path / "s" / "datasets" / n, encoding="utf-8").read().replace("\r\n", "\n")
        assert a == h[n].decode("utf-8").replace("\r\n", "\n")
    assert antes == {d: sorted(os.listdir(os.path.join(imagenes, *d.split("/")))) for d in antes}
    assert {n: open(os.path.join(RAIZ, "datasets", n), "rb").read() for n in h} == h
