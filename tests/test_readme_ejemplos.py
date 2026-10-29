"""Los ejemplos del README usan archivos que existen y los comandos de entrada/lectura funcionan."""
import os
import re
import shlex
import subprocess
import sys

import pytest

from conftest import RAIZ

README = os.path.join(RAIZ, "README.md")
FLAGS_DE_ENTRADA = ("--csv", "--imagen", "--migrar-modelo")


def comandos():
    texto = open(README, encoding="utf-8").read()
    out = []
    for bloque in re.findall(r"```bash\n(.*?)```", texto, re.S):
        for linea in bloque.splitlines():
            linea = linea.split("#")[0].strip()
            if linea.startswith("python "):
                out.append(shlex.split(linea))
    return out


def test_hay_ejemplos_en_el_readme():
    assert len(comandos()) >= 15


@pytest.mark.parametrize("cmd", comandos(), ids=lambda c: " ".join(c)[:70])
def test_los_archivos_de_entrada_y_scripts_existen(cmd):
    if cmd[1] == "-m":
        return
    assert os.path.isfile(os.path.join(RAIZ, cmd[1])), f"falta el script {cmd[1]}"
    for i, tok in enumerate(cmd):
        if tok in FLAGS_DE_ENTRADA:
            assert os.path.isfile(os.path.join(RAIZ, cmd[i + 1])), f"el ejemplo usa {cmd[i + 1]}, que no existe"


@pytest.mark.parametrize("script", sorted({c[1] for c in comandos() if c[1] != "-m"}))
def test_cada_script_del_readme_responde_a_help(script):
    p = subprocess.run([sys.executable, os.path.join(RAIZ, script), "--help"], capture_output=True, text=True, cwd=RAIZ)
    assert p.returncode == 0 and "usage" in p.stdout.lower()


SOLO_LECTURA = [c for c in comandos()
                if c[1] in ("CarGross.py", "probar_imagen.py")
                and not any(f in c for f in ("--guardar-modelo", "--guardar-aprendizaje", "--aprender"))]


@pytest.mark.parametrize("cmd", SOLO_LECTURA, ids=lambda c: " ".join(c)[:70])
def test_los_ejemplos_de_lectura_terminan_bien(cmd, tmp_path):
    extra = ["--salida-grafico", str(tmp_path / "d.png")] if cmd[1] == "probar_imagen.py" else []
    p = subprocess.run([sys.executable] + cmd[1:] + extra, capture_output=True, text=True, cwd=RAIZ)
    assert p.returncode == 0, p.stdout[-400:] + p.stderr[-400:]


def test_el_readme_no_promete_archivos_inexistentes():
    texto = open(README, encoding="utf-8").read()
    rutas = set(re.findall(r"`((?:docs|resultados|config|datasets_v2|scripts|evaluacion|tests)/[\w./-]+\.(?:md|py|csv|json))`", texto))
    generadas = {"resultados/comparacion.md", "resultados/ablacion.csv", "resultados/barrido_orden.md",
                 "resultados/diagnostico_fondo.md", "resultados/baseline_original.md", "resultados/mediciones/"}
    faltan = [r for r in rutas if not os.path.exists(os.path.join(RAIZ, r))]
    assert not faltan, f"el README menciona archivos que no existen: {faltan} (generados esperables: {generadas})"
