"""probar_imagen.py e inferencia.py: inferencia sin cambio de pesos, aprendizaje
explicito en memoria, guardado en archivos nuevos y codigos de salida."""
import hashlib
import json
import os

import cv2
import numpy as np
import pytest

import extraccion as X
import inferencia
import probar_imagen
from CarGross import RedART1, cargar_modelo
from conftest import RAIZ
from validacion import COD_ENTRADA, COD_OK, COD_USO


def foto(tmp_path, nombre="fruta.png"):
    img = np.full((240, 320, 3), (215, 225, 230), np.uint8)
    cv2.circle(img, (160, 125), 70, (40, 60, 200), -1)
    ruta = str(tmp_path / nombre)
    cv2.imwrite(ruta, img)
    return ruta, img


def hash_archivo(ruta):
    return hashlib.sha256(open(ruta, "rb").read()).hexdigest()


def crear_modelos(carpeta, img, forzar_novedad=True):
    """Modelos v2 minimos con el preprocesamiento legado. Si forzar_novedad, el prototipo es el
    complemento del vector de la foto: ninguna categoria resuena."""
    pm, pc = X.preprocesamiento_legado("morfologia"), X.preprocesamiento_legado("calidad")
    med = X.medir_foto(img, segmentador="v1")
    vm, vc = X.vectorizar(med, pm), X.vectorizar(med, pc)
    os.makedirs(carpeta, exist_ok=True)
    rutas = {}
    for etapa, pre, v, etiq in (("morfologia", pm, vm, "citrico"), ("calidad", pc, vc, "premium")):
        red = RedART1(len(v), 0.8)
        proto = [1 - b for b in v] if forzar_novedad else list(v)
        red.top_down = [proto]
        meta = {"cols_features": list(pre["cols_features"]),
                "mapeo_clusters": {"0": {"n": 5, "etiqueta": etiq, "estado": "etiquetada", "pureza": 0.8}}}
        ruta = os.path.join(carpeta, f"art1_{etapa}.json")
        red.guardar(ruta, meta, preprocesamiento=pre)
        rutas[etapa] = ruta
    return rutas, vm


def test_inferir_no_modifica_pesos_ni_archivos(tmp_path):
    ruta, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    m1, m2 = cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"])
    huellas = (m1.red.huella(), m2.red.huella())
    h_archivos = {k: hash_archivo(v) for k, v in rutas.items()}
    r = inferencia.inferir_imagen(img, m1, m2)
    assert (m1.red.huella(), m2.red.huella()) == huellas and r["cambios"] == []
    assert not r["etapa1"]["resuena"] and r["etapa1"]["etiqueta"] is None and r["etapa2"] is None
    assert {k: hash_archivo(v) for k, v in rutas.items()} == h_archivos


def test_la_novedad_no_recibe_etiqueta_ni_deriva_a_calidad(tmp_path):
    _, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    r = inferencia.inferir_imagen(img, cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"]))
    assert r["derivada"] is False and r["etapa1"]["motivo"] == "vigilancia"


def test_resultado_expone_coincidencia_y_pureza_con_su_significado(tmp_path):
    _, img = foto(tmp_path)
    rutas, vm = crear_modelos(tmp_path / "mod", img, forzar_novedad=False)
    r = inferencia.inferir_imagen(img, cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"]))
    e1 = r["etapa1"]
    assert e1["resuena"] and e1["coincidencia"] == 1.0 and e1["etiqueta"] == "citrico"
    assert e1["pureza_entrenamiento"] == 0.8 and e1["n_entrenamiento"] == 5
    assert "coincidencia" in e1 and "confianza" not in e1 and "probabilidad" not in e1


def test_aprender_crea_categoria_sin_etiqueta_en_memoria(tmp_path):
    _, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    m1, m2 = cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"])
    r = inferencia.inferir_imagen(img, m1, m2, aprender=True, nombre_imagen="fruta.png")
    assert len(m1.red.top_down) == 2 and r["cambios"][0]["evento"] == "categoria_nueva"
    nueva = m1.categoria(1)
    assert nueva["etiqueta"] is None and nueva["estado"] == "nueva_sin_etiqueta"      # etiqueta no inventada
    assert m1.meta["historial_aprendizaje"][0]["imagen"] == "fruta.png"
    # una segunda presentacion resuena con la categoria nueva, que sigue sin etiqueta: no se deriva a calidad
    r2 = inferencia.inferir_imagen(img, m1, m2)
    assert r2["etapa1"]["resuena"] and r2["etapa1"]["cluster"] == 1 and r2["etapa1"]["etiqueta"] is None
    assert r2["derivada"] is False


def test_guardar_aprendizaje_usa_archivo_nuevo_y_no_pisa_el_original(tmp_path):
    _, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    original = hash_archivo(rutas["morfologia"])
    m1, m2 = cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"])
    inferencia.inferir_imagen(img, m1, m2, aprender=True)
    destino = inferencia.guardar_aprendizaje(m1, "morfologia", str(tmp_path / "nuevo"))
    assert os.path.dirname(destino) == str(tmp_path / "nuevo") and hash_archivo(rutas["morfologia"]) == original
    guardado = cargar_modelo(destino)
    assert len(guardado.red.top_down) == 2 and guardado.categoria(1)["estado"] == "nueva_sin_etiqueta"
    with pytest.raises(FileExistsError):                                         # tampoco pisa el nuevo
        inferencia.guardar_aprendizaje(m1, "morfologia", str(tmp_path / "nuevo"))
    with pytest.raises(Exception, match="modelo de origen"):                     # ni su propio origen
        inferencia.guardar_aprendizaje(m1, "morfologia", str(tmp_path / "mod"))


def test_aprender_con_resonancia_actualiza_prototipo_y_lo_registra(tmp_path):
    _, img = foto(tmp_path)
    rutas, vm = crear_modelos(tmp_path / "mod", img, forzar_novedad=False)
    m1, m2 = cargar_modelo(rutas["morfologia"]), cargar_modelo(rutas["calidad"])
    m1.red.top_down = [vm[:-1] + [1 - vm[-1]]] if len(vm) else m1.red.top_down        # prototipo casi igual
    m1.red.rho = 0.5
    prototipo = list(m1.red.top_down[0])
    r = inferencia.inferir_imagen(img, m1, m2, aprender=True)
    c = r["cambios"][0]
    assert c["evento"] in ("prototipo_actualizado", "sin_cambio")
    assert all(a <= b for a, b in zip(m1.red.top_down[0], prototipo))               # AND: nunca gana unos


def test_modelo_legado_no_se_puede_guardar_sin_migrar(tmp_path):
    m = cargar_modelo(os.path.join(RAIZ, "modelos", "art1_morfologia.json"))
    with pytest.raises(Exception, match="formato 1"):
        inferencia.guardar_aprendizaje(m, "morfologia", str(tmp_path))


# --- CLI de probar_imagen ---------------------------------------------------------------------
def correr(argv, capsys):
    rc = probar_imagen.main(argv)
    o = capsys.readouterr()
    return rc, o.out, o.err


def test_cli_inferencia_normal_y_pesos_sin_cambios(tmp_path, capsys):
    ruta, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img, forzar_novedad=False)
    rc, out, _ = correr(["--imagen", ruta, "--modelo-morfologia", rutas["morfologia"],
                         "--modelo-calidad", rutas["calidad"], "--sin-grafico"], capsys)
    assert rc == COD_OK and "Pesos sin cambios" in out and "NO es probabilidad de acierto" in out
    assert "NO es la confianza de esta foto" in out and "no hay rechazo de objetos ajenos" in out


def test_cli_aprender_explica_y_guarda_solo_si_se_pide(tmp_path, capsys):
    ruta, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    h = {k: hash_archivo(v) for k, v in rutas.items()}
    base = ["--imagen", ruta, "--modelo-morfologia", rutas["morfologia"], "--modelo-calidad", rutas["calidad"],
            "--sin-grafico", "--aprender"]
    rc, out, _ = correr(base, capsys)
    assert rc == COD_OK and "APRENDIZAJE ACTIVADO" in out and "SIN etiqueta" in out
    assert "NO se guardaron" in out and not os.path.exists(tmp_path / "nuevo")
    rc, out, _ = correr(base + ["--guardar-aprendizaje", str(tmp_path / "nuevo")], capsys)
    assert rc == COD_OK and os.path.isfile(tmp_path / "nuevo" / "art1_morfologia.json")
    assert {k: hash_archivo(v) for k, v in rutas.items()} == h                   # originales intactos
    rc, _, err = correr(base + ["--guardar-aprendizaje", str(tmp_path / "nuevo")], capsys)
    assert rc == COD_USO and "ya existe" in err
    rc, _, err = correr(base + ["--guardar-aprendizaje", str(tmp_path / "mod")], capsys)
    assert rc == COD_ENTRADA and "modelo de origen" in err


def test_cli_codigos_de_salida_y_ayuda(tmp_path, capsys):
    rc, _, err = correr(["--imagen", str(tmp_path / "no.png")], capsys)
    assert rc == COD_ENTRADA and "no se encontro el archivo de imagen" in err and "--help" in err
    malo = tmp_path / "roto.png"
    malo.write_bytes(b"no es una imagen")
    rc, _, err = correr(["--imagen", str(malo), "--legado"], capsys)
    assert rc == COD_ENTRADA and "no pudo decodificar" in err
    ruta, img = foto(tmp_path)
    rc, _, err = correr(["--imagen", ruta, "--modelo-morfologia", str(tmp_path / "no.json")], capsys)
    assert rc == COD_ENTRADA and "No se entrena nada en silencio" in err
    rc, _, err = correr(["--imagen", ruta, "--guardar-aprendizaje", str(tmp_path / "x")], capsys)
    assert rc == COD_USO and "requiere --aprender" in err
    with pytest.raises(SystemExit) as e:
        probar_imagen.main(["--ayuda"])
    assert e.value.code == 0 and "--aprender" in capsys.readouterr().out


def test_cli_modelo_corrupto_codigo_3(tmp_path, capsys):
    ruta, img = foto(tmp_path)
    rutas, _ = crear_modelos(tmp_path / "mod", img)
    d = json.load(open(rutas["morfologia"], encoding="utf-8"))
    d["top_down"][0] = [2] * len(d["top_down"][0])
    with open(rutas["morfologia"], "w", encoding="utf-8") as f:
        json.dump(d, f)
    rc, _, err = correr(["--imagen", ruta, "--modelo-morfologia", rutas["morfologia"],
                         "--modelo-calidad", rutas["calidad"], "--sin-grafico"], capsys)
    assert rc == COD_ENTRADA and "vector binario" in err


def test_cli_legado_con_modelos_entregados_y_ejemplo_incluido(capsys, tmp_path):
    ejemplo = os.path.join(RAIZ, "ejemplos", "naranja.jpg")
    rc, out, _ = correr(["--imagen", ejemplo, "--legado", "--salida-grafico", str(tmp_path / "d.png")], capsys)
    assert rc == COD_OK and "formato 1" in out and os.path.isfile(tmp_path / "d.png")
    assert "modelo en formato 1" in out


def test_legado_usa_el_orden_de_variables_y_rho_entregados():
    img = cv2.imread(os.path.join(RAIZ, "ejemplos", "naranja.jpg"))
    r = inferencia.inferir_imagen(img, cargar_modelo(os.path.join(RAIZ, "modelos", "art1_morfologia.json")),
                                  cargar_modelo(os.path.join(RAIZ, "modelos", "art1_calidad.json")))
    assert r["etapa1"]["cols_features"] == list(X.preprocesamiento_legado("morfologia")["cols_features"])
    assert r["etapa1"]["rho"] == 0.85 and any("formato 1" in a for a in r["advertencias"])
